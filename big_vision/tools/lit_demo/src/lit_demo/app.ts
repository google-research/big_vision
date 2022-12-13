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
