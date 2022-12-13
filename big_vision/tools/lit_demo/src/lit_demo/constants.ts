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
 * @fileoverview Project-wide constants.
 */

// Can be overwritten with setBaseUrl() below.
// let baseUrl = 'https://google-research.github.io/vision_transformer/lit';
let baseUrl = 'https://figur.li/jax2tfjs';
// Can be overwritten with setModels() below.
let models = ['tiny', 'small'];

/** Allows to set abnew base URL. ase URL on which all other.  */
export const setBaseUrl = (newBaseUrl: string) => {
  baseUrl = newBaseUrl;
};

/** Retrieves URL for a model-specific file (vocabulary, embeddings, ...). */
export const getModelFileUrl = (name: string, relativePath: string) => (
  `${baseUrl}/data/models/${name}/${relativePath}`
);

/** Retrieves the URL for images information JSON file. */
export const getImagesInfoUrl = () => `${baseUrl}/data/images/info.json`;

/** Retrieves the URL for an image. */
export const getImageUrl = (id: string) => `${baseUrl}/data/images/${id}.jpg`;

/** Returns names of available models. */
export const getModels = () => models;

/** Sets names of available models. */
export const setModels = (newModels: string[]) => {
  models = newModels;
};
