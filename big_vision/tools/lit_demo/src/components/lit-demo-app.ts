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
 * @fileoverview Main application.
 */

import {html, LitElement} from 'lit';

import {app} from '../lit_demo/app';
import {parseUrl, State} from '../lit_demo/url_utils';

import './image-carousel';
import {ImagePrompts} from './image-prompts';
import './loading-animation';
import {MessageList} from './message-list';
import {ModelControls} from './model-controls';

import {customElement, property, query} from 'lit/decorators.js';
import styles from './lit-demo-app.scss';

/**
 * Main application container.
 */
@customElement('lit-demo-app')
export class LitDemoApp extends LitElement {

  static override styles = [styles];

  @property({type: Boolean})
  loading: boolean = true;

  @query('message-list')
  messageList!: MessageList;
  @query('model-controls')
  modelControls!: ModelControls;
  @query('#examples')
  examples!: HTMLElement;

  state?: State;
  lingeringWarning?: string;

  constructor() {
    super();
    window.onerror = this.onglobalerror.bind(this);
    this.load();
  }

  onglobalerror(message: string|Event, source: string|undefined, lineno: number|undefined) {
    source = source || '';
    source = source.substring(source.lastIndexOf('/') + 1);
    this.messageList.error(
      `<b>Javascript error</b> at ${source}:${lineno}<br>` +
      `<code>${message}</code>`,
      {rawHtml: true});
  }

  async load() {
    await app.load();
    this.loading = false;
    try {
      this.state = parseUrl();
    } catch (error) {
      this.messageList.warning(`Could not parse URL: ${error}`);
    }
  }

  override updated() {
    if (this.state && this.examples) {
      this.modelControls.setModel(this.state.modelName);
      this.addFromState(this.state);
      this.state = undefined;
    }
  }

  override render() {
    return html`
      ${this.loading ? html`` : html`
        <image-carousel @image-select=${this.onImageSelect}></image-carousel>
        <model-controls></model-controls>
      `}
      <message-list></message-list>
      ${this.loading ? html`
      <div class="loading-container">
          <loading-animation title="loading..."></loading-animation>
        </div>
      ` : html`
        <div id="examples">
        </div>
      `}
    `;
  }

  onImageSelect(event: CustomEvent) {
    this.addImagePrompts(event.detail.id);
  }

  addFromState(state: State) {
    const imagePrompts = new ImagePrompts(state.imageId);
    imagePrompts.setPrompts(state.prompts);
    this.examples.insertBefore(imagePrompts, this.examples.childNodes[0]);
  }

  addImagePrompts(id: string): ImagePrompts {
    const imagePrompts = new ImagePrompts(id);
    imagePrompts.addEventListener('duplicate', (event: Event) => {
      const duplicated = this.addImagePrompts(id);
      duplicated.setPrompts(imagePrompts.getPrompts());
    });
    this.examples.insertBefore(imagePrompts, this.examples.childNodes[0]);
    return imagePrompts;
  }
}
