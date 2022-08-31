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
