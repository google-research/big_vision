/**
 * @fileoverview Global app state.
 */

import {ImageData} from './data';
import {Models} from './compute';

/**
 * Container class holding image data and models.
 *
 * The main application component would typically call `load()` and then show
 * the components depending on this class asynchronously.
 */
export class App {

  imageData = new ImageData();
  models = new Models();

  ready: boolean = false;

  async load() {
    await this.imageData.load();
    this.ready = true;
  }
}

/**
 * Global app state.
 */
export const app = new App();
