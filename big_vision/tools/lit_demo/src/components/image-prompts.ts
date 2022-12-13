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
 * @fileoverview Image and text prompts.
 */

import {html, LitElement} from 'lit';
import * as naughtyWords from 'naughty-words';

import {app} from '../lit_demo/app';
import {getBackend} from '../lit_demo/compute';
import {getImageUrl} from '../lit_demo/constants';
import {getUrl} from '../lit_demo/url_utils';

import {MessageList} from './message-list';

import {customElement, query} from 'lit/decorators.js';
import styles from './image-prompts.scss';

const setHref = (anchorEl: HTMLAnchorElement, href:string) => {
  anchorEl. href = href;
};

const HTML_TEMPLATE = `
    We cannot include the word "{word}" as it is found on the list
    <a target="_blank" href="https://github.com/LDNOOBW/naughty-words-js/blob/master/{lang}.json">naughty-words/{lang}</a>.
    We understand blocklists are an imperfect solution but we believe it's
    important to ensure these models are not misused, and hope that in this
    instance it does not serve to marginalise anybody. If you don't agree,
    please reach out via
    <a target="_blank" href="https://forms.gle/5eTsjMzXSY8qzJjB7">form link</a>.
`;


/**
 * Shows image and text prompts, and computes similarities.
 *
 * Also dispatches some events like `'duplicate'` to the parent.
 */
@customElement('image-prompts')
export class ImagePrompts extends LitElement {

  static override styles = [styles];

  @query('message-list')
  messageList!: MessageList;
  @query('.animation')
  animation!: HTMLElement;
  @query('.bottom')
  bottom!: HTMLElement;

  lastPrompts?: string[];

  constructor(private readonly imageId: string) {
    super();
  }

  override firstUpdated() {
    if (getBackend() !== 'webgl') {
      this.messageList.warning(
          'Please activate WebGL. Running ML demos on ' +
          'CPU will drain your battery in no time...');
    }
  }

  onDuplicate() {
    this.dispatchEvent(new Event('duplicate'));
  }

  onRemove() {
    this.remove();
  }

  onClear() {
    this.shadowRoot!.querySelectorAll('.prompt').forEach((input: Element) => {
      (input as HTMLInputElement).value = '';
    });
    (this.shadowRoot!.querySelector('.prompt') as HTMLInputElement).focus();
  }

  onKeyup(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      this.onCompute();
    }
  }

  async setPrompts(prompts: string[]) {
    await this.updateComplete;
    this.shadowRoot!.querySelectorAll('.prompt').forEach((input: Element, idx: number) => {
        (input as HTMLInputElement).value = prompts[idx] || '';
    });
  }

  getPrompts(): string[] {
    return [...this.shadowRoot!.querySelectorAll('.prompt')].map((input: Element) =>
        (input as HTMLInputElement).value
    );
  }

  override render() {
    const row = app.imageData.get(this.imageId);
    const inputs = row.prompts.split(',').map((prompt: string, idx: number) => {
      return html`
        <div class="item">
          <div class="pct"></div>
          <input @keyup=${this.onKeyup} class="prompt" placeholder=${`Prompt #${idx + 1}`} value=${prompt}>
          <div class="bar"></div>
        </div>
      `;
    });
    return html`
      <div class="image-prompt">
        <div class="left">
          <div class="wrapper">
            <img src="${getImageUrl(this.imageId)}">
            <a target="_blank" href="${row.source}" class="src">source</a>
          </div>
          <div class="animation">
            <div class="computing">✨✨Computing✨✨</div>
          </div>
        </div>
        <div class="right">
          <message-list></message-list>
          <div class="buttons">
            <button @click=${this.onDuplicate} title="Duplicate example">Duplicate</button>
            <button @click=${this.onRemove} title="Remove example">Remove</button>
            <button @click=${this.onClear} title="Clear inputs">Clear</button>
            <button @click=${this.onCompute} title="Compute embeddings (same as pressing enter)">🔥 Compute 🔥</button>
          </div>
          ${inputs}
          <div class="bottom">
            <span>Model: <span class="model">?</span></span>
            <a href="#" class="tweet" target="_blank">tweet</a>
          </div>
        </div>
      </div>
    `;
  }

  onCompute() {
    if (!app.models.ready) {
      this.messageList.warning('Model not ready yet.');
      return;
    }

    const model = app.models.model!;
    const zimgIdx = model.zimgIds!.indexOf(this.imageId);
    if (zimgIdx === -1) {
      this.messageList.warning('Model is missing this image embedding');
      return;
    }

    const texts = this.getPrompts();
    for (const text of texts) {
      for (const word of text.toLocaleLowerCase().split(/\s+/g)) {
        // tslint:disable-next-line:ban-module-namespace-object-escape
        for (const lang of Object.keys(naughtyWords)) {
          if (lang === 'default') {
            continue;
          }
          // tslint:disable-next-line:ban-module-namespace-object-escape
          const words = (naughtyWords as {[key: string]: string[]})[lang];
          if (words.indexOf(word) !== -1) {
            const msg = HTML_TEMPLATE.replace(/\{word\}/g, word).replace(/\{lang\}/g, lang);
            this.messageList.warning(msg, {rawHtml: true});
            return;
          }
        }
      }
    }

    const compute = () => {
      let probs: number[]|undefined;
      try {
        // ??? how to move into webworker (to avoid freezing UI) ?
        // https://github.com/tensorflow/tfjs/issues/102
        probs = model.computeProbabilities(texts, zimgIdx);
      } catch (error) {
        if ((error as Error).message.toLocaleLowerCase().match(/greater than .* maximum/)) {
          this.messageList.warning('Model too large for Browser!');
          return;
        }
        throw error;
      }
      this.setProbabilities(probs);
      this.lastPrompts = this.getPrompts();
      this.animation.style.opacity = '0';
    };

    this.animation.style.opacity = '1';
    this.messageList.clear();
    setTimeout(compute, 10);  // Give UI some time to update.
  }

  setProbabilities(probs: number[]) {
    const pcts = [...this.shadowRoot!.querySelectorAll('.pct')] as HTMLElement[];
    const bars = [...this.shadowRoot!.querySelectorAll('.bar')] as HTMLElement[];
    this.hideBottom();
    for(let i = 0; i < Math.max(probs.length, pcts.length, bars.length); i++) {
      const prob = probs[i] || 0;
      const pct = `${Math.round(prob * 1e3) / 1e1}%`;
      bars[i].style.width = pct;
      if (prob) {
        pcts[i].innerText = pct;
        pcts[i].style.opacity = '1';
      } else {
        pcts[i].style.opacity = '0';
      }
    }
    this.updateBottom();
  }

  updateBottom() {
    const tweet = this.shadowRoot!.querySelector('.tweet') as HTMLAnchorElement;
    const url = getUrl(app.models.model!.name, this.imageId, this.getPrompts());
    const description = app.imageData.get(this.imageId).description;
    const text = `LiT matching prompts to an image of "${description}"\n\n#lit_demo\n`;
    setHref(tweet, 'https://twitter.com/intent/tweet' +
        '?url=' + encodeURIComponent(url) +
        '&text=' + encodeURIComponent(text));
    this.bottom.style.opacity = '1';
    const model = this.shadowRoot!.querySelector('.model') as HTMLAnchorElement;
    model.innerText = app.models.model!.name;
  }

  hideBottom() {
    this.bottom.style.opacity = '0';
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'image-prompts': ImagePrompts;
  }
}
