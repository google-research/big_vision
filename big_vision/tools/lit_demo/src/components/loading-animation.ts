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
