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

import {app} from '../lit_demo/app';
import {getImageUrl} from '../lit_demo/constants';
import {ImageRow} from '../lit_demo/data';

import {customElement} from 'lit/decorators.js';
import styles from './image-carousel.scss';

/**
 * Shows multiple images in a horizontal carousel.
 *
 * Dispatches `'image-select'` event when an image is clicked/tapped.
 */
@customElement('image-carousel')
export class ImageCarousel extends LitElement {
  static override styles = [styles];

  onClick(id: string) {
    const event =
        new CustomEvent('image-select', {composed: true, detail: {id}});
    this.dispatchEvent(event);
  }

  override render() {
    const images = app.imageData.rows.map(
        (row: ImageRow) => html`
          <div class="thumb">
            <img @click=${() => {
          this.onClick(row.id);
        }} data-id=${row.id} src="${getImageUrl(row.id)}">
          </div>
        `);
    return html`
      <div class="selector">
        <div class="inner">
          ${images}
        </div>
      </div>
      <p>Select an image 👆 to get started.</p>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'image-carousel': ImageCarousel;
  }
}
