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
 * @fileoverview A list of dismissable info/warning/error messages.
 */

import {html, LitElement} from 'lit';

import {unsafeHTML} from 'lit/directives/unsafe-html.js';

import {customElement} from 'lit/decorators.js';
import styles from './message-list.scss';

enum MessageType {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
}

interface Message {
  message: string;
  type: MessageType;
  rawHtml: boolean;
}


/**
 * Shows info/warning/error messages that remain until closed by user.
 */
@customElement('message-list')
export class MessageList extends LitElement {
  static override styles = [styles];

  messages: Message[] = [];

  addMessage(message: Message) {
    this.messages.push(message);
    this.requestUpdate();
  }

  info(message: string, {rawHtml = false}: {rawHtml?: boolean} = {}) {
    this.addMessage({message, type: MessageType.INFO, rawHtml});
  }

  warning(message: string, {rawHtml = false}: {rawHtml?: boolean} = {}) {
    this.addMessage({message, type: MessageType.WARNING, rawHtml});
  }

  error(message: string, {rawHtml = false}: {rawHtml?: boolean} = {}) {
    this.addMessage({message, type: MessageType.ERROR, rawHtml});
  }

  removeMessage(event: Event, idx: number) {
    this.messages.splice(idx, 1);
    (event.target! as HTMLElement).closest('.message')!.remove();
  }

  clear() {
    this.messages = [];
    while (this.firstChild) this.firstChild.remove();
  }

  override render() {
    return this.messages.map(
        (message: Message, idx: number) => html`
      <div class="${message.type} message">
        <span class="label">${
            message.rawHtml ? unsafeHTML(message.message) :
                              message.message}</span>
        <span @click=${(e: Event) => {
          this.removeMessage(e, idx);
        }} class="close">✖</span>
      </div>
    `);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'message-list': MessageList;
  }
}
