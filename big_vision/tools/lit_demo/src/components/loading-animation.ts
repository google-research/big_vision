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
 * @fileoverview Carousel of images.
 */

import {html, LitElement} from 'lit';

import {customElement} from 'lit/decorators.js';
import styles from './loading-animation.scss';

/**
 * Shows an animated loading animation.
 */
@customElement('loading-animation')
export class LoadingAnimation extends LitElement {

  static override styles = [styles];

  override render() {
    return html`
      <div class="lds-ellipsis">
        <div></div>
        <div></div>
        <div></div>
        <div></div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'loading-animation': LoadingAnimation;
  }
}
