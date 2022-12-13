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

// Copied & adapted from
// https://github.com/tensorflow/tfjs-models/blob/master/universal-sentence-encoder/src/tokenizer/index.ts

import {TOKEN_SEPARATOR, stringToChars, Tokenizer as TokenizerInterface, Vocabulary} from './common';
import {Trie} from './trie';

function processInput(str: string): string {
  const normalized = str.normalize('NFKC');
  return normalized.length > 0 ?
      TOKEN_SEPARATOR + normalized.replace(/ /g, TOKEN_SEPARATOR) :
      normalized;
}

// The first tokens are reserved for unk, control symbols, and user-defined
// symbols.
const RESERVED_SYMBOLS_COUNT = 6;

interface Score {
  key: string[];
  score: number;
  index: number;
}

/**
 * Sentencepiece tokenizer implementing the UNIGRAM algorithm.
 *
 * `Tokenizer.encode()` is a port of `EncodeAsIds` from the SentencePiece
 * library (https://github.com/google/sentencepiece). Encode uses the Viterbi
 * algorithm to find the most likely sequence of tokens that comprise the input.
 * For more details, refer to https://arxiv.org/pdf/1804.10959.pdf.
 */
export class Tokenizer implements TokenizerInterface {
  trie: Trie;

  constructor(
      private readonly vocabulary: Vocabulary,
      private readonly reservedSymbolsCount = RESERVED_SYMBOLS_COUNT) {
    this.trie = new Trie();

    for (let i = this.reservedSymbolsCount; i < this.vocabulary.length; i++) {
      this.trie.insert(this.vocabulary[i][0], this.vocabulary[i][1], i);
    }
  }

  encode(input: string): number[] {
    const nodes: Array<{[index: number]: Score[]}> = [];
    const words: number[] = [];
    const best: number[] = [];

    input = processInput(input);

    const symbols = stringToChars(input);

    for (let i = 0; i <= symbols.length; i++) {
      nodes.push({});
      words.push(0);
      best.push(0);
    }

    // Construct the lattice.
    for (let i = 0; i < symbols.length; i++) {
      const matches = this.trie.commonPrefixSearch(symbols.slice(i));

      for (let j = 0; j < matches.length; j++) {
        const piece = matches[j];
        const obj = {key: piece[0], score: piece[1], index: piece[2]};

        const endPos = piece[0].length;
        if (nodes[i + endPos][i] == null) {
          nodes[i + endPos][i] = [];
        }

        nodes[i + endPos][i].push(obj);
      }
    }

    for (let endPos = 0; endPos <= symbols.length; endPos++) {
      for (const startPos in nodes[endPos]) {
        if (!nodes[endPos].hasOwnProperty(startPos)) continue;
        const arr = nodes[endPos][startPos];

        for (let j = 0; j < arr.length; j++) {
          const word = arr[j];
          const score = word.score + best[endPos - word.key.length];

          if (best[endPos] === 0 || score >= best[endPos]) {
            best[endPos] = score;
            words[endPos] = arr[j].index;
          }
        }
      }
    }

    const results: number[] = [];

    // Backward pass.
    let iter = words.length - 1;
    while (iter > 0) {
      results.push(words[iter]);
      iter -= this.vocabulary[words[iter]][0].length;
    }

    // Merge consecutive unks.
    const merged = [];
    let isPreviousUnk = false;
    for (let i = 0; i < results.length; i++) {
      const id = results[i];
      if (!(isPreviousUnk && id === 0)) {
        merged.push(id);
      }

      isPreviousUnk = id === 0;
    }

    return merged.reverse();
  }
}
