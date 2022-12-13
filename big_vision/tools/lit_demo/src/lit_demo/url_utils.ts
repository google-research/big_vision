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
 * @fileoverview (De)serialize state from/to URL.
 */

// Should be updated whenever URLs are not compatible anymore
// (e.g. adding new images)
export const VERSION = 'v2';
// version history:
// v1 used row number instead of image id

const V1_IMAGE_IDS = [
  '1',  '48', '43', '22', '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',
  '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
  '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
  '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47',
  '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60'
];

/**
 * State that can be stored in the URL.
 */
export interface State {
  /** Name of the model. */
  modelName: string;
  /** ID Of the image. */
  imageId: string;
  /** List of text prompts. */
  prompts: string[];
}

/**
 * Returns a URL for provided model/image/prompts.
 */
export const getUrl =
    (modelName: string, imageId: string, prompts: string[]): string => {
      let href = window.location.href;
      if (href.indexOf('#') !== -1) {
        href = href.substring(0, href.indexOf('#'));
      }
      const parts = [
        VERSION,
        modelName,
        imageId,
        ...prompts,
      ];
      return href + '#' + parts.map(encodeURIComponent).join('|');
    };

/**
 * Parses an URL and returns a `State`, or undefined if no state is spefified.
 *
 * Raises an exception if there was a problem with the parsing of the URL.
 */
export const parseUrl = (): State|undefined => {
  const hash = window.location.hash.substring(1);
  if (!hash) return;
  const parts = hash.split(/\|/g);
  if (parts.length < 4) {
    throw new Error(`Invalid URL: "${hash}"`);
  }
  let [version, modelName, imageId, ...texts] = parts;
  if (version === VERSION) {
  } else if (version === 'v1') {
    const idx = Number(imageId);
    if (isNaN(idx)) throw new Error(`Expected idx="${idx}" to be numerical!`);
    imageId = V1_IMAGE_IDS[idx];
  } else {
    throw new Error(`Incompatible version: ${version} (supported: ${VERSION})`);
  }
  return {
    modelName,
    imageId,
    prompts: texts.map(decodeURIComponent),
  };
};
