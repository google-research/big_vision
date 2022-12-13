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
 * @fileoverview Controls to choose model.
 */

import {html, LitElement} from 'lit';

import {getModels} from '../lit_demo/constants';
import {app} from '../lit_demo/app';

import {customElement, property} from 'lit/decorators.js';
import styles from './model-controls.scss';

/**
 * Shows controls for model selection, progress bar, and status text.
 */
@customElement('model-controls')
export class ModelControls extends LitElement {

  static override styles = [styles];

  @property({attribute: false})
  progress: number = 0;

  @property({attribute: false})
  status: string = 'Initializing...';

  constructor() {
    super();
    app.models.addListener(this.onModelUpdate.bind(this));
    app.models.load(getModels()[0]);
  }

  onModelUpdate(progress: number, message?: string) {
    this.progress = progress;
    if (message) this.status = message;
  }

  onModelChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    const name = target.value;
    app.models.load(name).catch((error) => {
      this.status = `ERROR loading model "${name}": ${error}`;
    });
  }

  async setModel(model: string) {
    if (getModels().indexOf(model) === -1) {
      throw new Error(`Model "${model}" not found!`);
    }
    await this.updateComplete;
    const dropdown = this.shadowRoot!.querySelector('#model_dropdown') as HTMLSelectElement;
    dropdown.value = model;
    dropdown.dispatchEvent(new Event('change'));
  }

  override render() {
    const options = getModels().map((model: string) =>
        html`<option value="${model}">${model}</option>`);
    return html`
      <div class="controls">
        <label for="model_dropdown">Model:</label>
        <select @change=${this.onModelChange} id="model_dropdown">
          ${options}
        </select>
        <progress value=${this.progress * 100} max=100></progress>
        <div class="status">${this.status}</div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'model-controls': ModelControls;
  }
}
