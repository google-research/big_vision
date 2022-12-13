/**
 * @license
 * Copyright Big Vision Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @fileoverview Model code.
 */

import '@tensorflow/tfjs-backend-webgl';

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import {MathBackendWebGL} from '@tensorflow/tfjs-backend-webgl';

import {getTokenizer, Tokenizer} from '../tokenizers/index';

import {getModelFileUrl} from './constants';

/**
 * Callback to be updated with model load status.
 *
 * @param progress: the callback function is repeatedly called with values from
 * 0 to 1 (both endpoints are guaranteed to be included)
 * @param message: optional message to be displayed to user
 */
export type StatusCallback = (progress: number, message?: string) => void;

const canonicalize = (s: string) => {
  s = s.toLocaleLowerCase();
  s = s.replace(/[^\w ]/g, '');
  s = s.replace(/\s+/g, ' ');
  return s.trim();
};

/**
 * The model definition is read from a JSON and specifies model details.
 */
// tslint:disable:enforce-name-casing
export interface ModelDefinition {
  /** Human-readable description of the model. */
  description: string;
  /** Tokenizer type. See ./tokenizers/index */
  tokenizer_type: string;
  /** Temperature for computing softmax. */
  temperature: number;
  /** Token used for padding. */
  pad_value: number;
  /** Maximum token length. */
  max_len: number;
  /** Dimensionality of image/text embeddings. */
  embedding_size: number;
}
// tslint:enable:enforce-name-casing

/**
 * TFJS model to compute text embeddings and similarities.
 */
export class Model {
  def?: ModelDefinition;
  tokenizer?: Tokenizer;
  model?: tfconv.GraphModel;
  /** Pre-computed image embeddings. */
  zimgs?: tf.Tensor;
  /** Pre-computed text embeddings. */
  ztxts?: tf.Tensor;
  /** IDs for pre-computed image embeddings. */
  zimgIds?: string[];
  /** Prompts for pre-computed text embeddings. */
  ztxtPrompts?: string[];
  /** Will be set to `true` when `load()` has completed successfully. */
  ready: boolean = false;

  /**
   * @param name: Name of the model to be loaded. Will be used to construct the
   * model URL. Note that the model must be loaded via calling `load()` before
   * it can be used.
   */
  constructor(public name: string) {
  }

  /**
   * Loads model, tokenizer, and pre-computed embeddings.
   */
  async load(callback?: StatusCallback) {
    this.def =
        await fetch(getModelFileUrl(this.name, 'def.json')).then(resp => {
          if (resp.ok) return resp.json();
          throw new Error(`Could not load model def: ${resp.status}`);
        });
    console.log('def', this.def);

    const tokenizer = fetch(getModelFileUrl(this.name, 'vocabulary.json'))
                          .then(resp => resp.json())
                          .then(
                              vocabulary => getTokenizer(
                                  this.def!.tokenizer_type, vocabulary));

    const model =
        tfconv.loadGraphModel(getModelFileUrl(this.name, 'tfjs/model.json'), {
          onProgress: (progress: number) => {
            callback && callback(progress);
          }
        });

    const fetchBin = async (name: string) => {
      const response = await fetch(getModelFileUrl(this.name, `${name}.bin`));
      const blob = await response.blob();
      const data = await new Promise(resolve => {
        const reader = new FileReader();
        reader.addEventListener('loadend', () => {
          resolve(reader.result);
        });
        reader.readAsArrayBuffer(blob);
      });
      const arr = new Float32Array(data as Iterable<number>);
      const n = arr.length / this.def!.embedding_size;
      return tf.tensor(arr, [n, this.def!.embedding_size]);
    };
    const fetchTxt = (name: string) =>
        fetch(getModelFileUrl(this.name, `${name}.txt`))
            .then(response => response.text())
            .then(text => text.split(/\n/g));

    [this.tokenizer,
     this.model,
     this.zimgs,
     this.ztxts,
     this.zimgIds,
     this.ztxtPrompts,
    ] =
        [
          await tokenizer,
          await model,
          await fetchBin('zimgs'),
          await fetchBin('ztxts'),
          await fetchTxt('zimgs'),
          await fetchTxt('ztxts'),
        ];
    this.ready = true;
    await this.warmup();
    if (callback) callback(1, 'Done.');
  }

  private async warmup() {
    if (getBackend() !== 'webgl') return;

    const webGLBackend = tf.backend() as MathBackendWebGL;
    tf.env().set('ENGINE_COMPILE_ONLY', true);
    const tokens = tf.zeros([5, this.def!.max_len], 'int32');
    const preCompileResults =
        this.model!.predict({inputs: tokens}) as tf.Tensor;
    webGLBackend.checkCompileCompletion();
    webGLBackend.getUniformLocations();

    tf.env().set('ENGINE_COMPILE_ONLY', false);
    const warmUpResults = this.model!.predict({inputs: tokens}) as tf.Tensor;
    await warmUpResults.data();

    preCompileResults.dispose();
    warmUpResults.dispose();
  }

  /**
   * Tokenizes strings with the model's tokenizer.
   */
  tokenize(texts: string[]): tf.Tensor {
    if (!this.ready) throw new Error('Cannot tokenize: not ready');
    const tokenize = (text: string) => {
      const maxLen = this.def!.max_len || 16;
      const tokens = this.tokenizer!.encode(text).slice(0, maxLen);
      // eos="sticky"
      const tokenEos = tf.tensor(
          [
            ...tokens,
            ...new Array(16 - tokens.length).fill(this.def!.pad_value),
          ],
          undefined, 'int32');
      return tokenEos;
    };
    return tf.stack(texts.map(tokenize));
  }

  /**
   * Computes embeddings for text tokenized via `tokenize()`.
   */
  embed(tokens: tf.Tensor): tf.Tensor {
    if (!this.ready) throw new Error('Cannot embed: not ready');
    return this.model!.execute({inputs: tokens}) as tf.Tensor;
  }

  /**
   * Computes similarities between specified prompts and images. Images are
   * referenced by their ID.
   */
  computeSimilarities(texts: string[], imgidxs: number[]) {
    if (!this.ready) throw new Error('Cannot compute similarities: not ready');
    texts = texts.map(canonicalize);
    const precomputed =
        texts
            .map(text => {
              const idx = this.ztxtPrompts!.indexOf(text);
              return idx === -1 ? null : tf.slice(this.ztxts!, idx, 1);
            })
            .filter((x: tf.Tensor|null) => !!x) as tf.Tensor[];
    console.log(texts.length, 'texts, ', precomputed.length, 'precomputed');
    const textEmbeddings = texts.length === precomputed.length ?
        tf.concat(precomputed) :
        this.embed(this.tokenize(texts));
    const imageEmbeddingsTransposed = tf.transpose(
        tf.concat(imgidxs.map(idx => tf.slice(this.zimgs!, idx, 1))));
    const sims = tf.matMul(textEmbeddings, imageEmbeddingsTransposed);
    sims.print();
    return sims;
  }

  /**
   * Computes probabilities between a set of prompts and a single image
   * (identified by its ID).
   */
  computeProbabilities(texts: string[], imgidx: number): number[] {
    const sims = this.computeSimilarities(texts, [imgidx]);
    const row = tf.squeeze(tf.slice(tf.transpose(sims), 0, 1));
    return [...tf.softmax(tf.mul(this.def!.temperature, row)).dataSync()];
  }
}

/**
 * Container that holds a set of models.
 */
export class Models {
  private readonly map = new Map<string, Model>();
  private readonly listeners = new Set<StatusCallback>();
  model?: Model;

  /**
   * Adds a listener to be updated about individual models' loading progress.
   */
  addListener(callback: StatusCallback) {
    this.listeners.add(callback);
  }

  /**
   * Updates all listeners wth `progress` and `message`.
   */
  onUpdate(progress: number, message?: string) {
    if (progress === 1) {
      message = `Loaded model "${this.model?.name}".`;
    }
    for (const callback of this.listeners) {
      callback(progress, message);
    }
  }

  /**
   * Loads model and sets `model` attribute when ready.
   */
  async load(name: string) {
    if (this.map.has(name)) {
      this.model = this.map.get(name);
      this.onUpdate(1, `Loaded "${name}".`);
      return;
    }
    this.onUpdate(0, 'Loading...');
    this.model = new Model(name);
    await this.model.load(this.onUpdate.bind(this));
    this.map.set(name, this.model);
  }

  /**
   * Whether model referenced by `model` attribute is ready.
   */
  get ready(): boolean {
    return !!this.model?.ready;
  }
}

/** Returns backend, such as "cpu" or "webgl". */
export function getBackend(): string {
  return tf.getBackend();
}
