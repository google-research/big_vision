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
 * @fileoverview Accessing additional data.
 */

import {getImagesInfoUrl} from './constants';

/**
 * Information about a single image.
 */
export interface ImageRow {
  /** Stable ID of the image. */
  id: string;
  /** Set of example prompts for this image. */
  prompts: string;
  /** License of the image. */
  license: string;
  /** Where the image was originally downloaded from. */
  source: string;
  /** Short description of image. */
  description: string;
}
/**
 * Contains information about all images.
 */
export class ImageData {

  rows: ImageRow[] = [];
  /** Will be set to `true` when `load()` finishes. */
  ready = false;

  /**
   * Gets an image by ID. Throws an error if image is not found, data is not
   * loaded, or ID is not unique.
   */
  get(id: string): ImageRow {
    if (!this.ready) {
      throw new Error('ImageData not loaded!');
    }
    const matching = this.rows.filter(row => row.id === id);
    if (matching.length !== 1) {
      throw new Error(`Got unexpected ${matching.length} matches for id="${id}"`);
    }
    return matching[0];
  }

  /**
   * Loads image data asynchronously.
   */
  async load() {
    this.rows = (
      await fetch(getImagesInfoUrl())
      .then(response => {
        console.log('response', response);
        return response.json();
      })
    );
    this.ready = true;
  }
}
