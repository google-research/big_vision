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
