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

import 'jasmine';

describe('sentencepiece bpe test', () => {
  it('computes a thing when asked', () => {});
});

import * as bpe from './sentencepiece_bpe';
import {TOKEN_SEPARATOR, Vocabulary} from './common';

const vocab: Vocabulary = [
  [TOKEN_SEPARATOR, 0],  // 0
  ['a', 0],              // 1
  ['e', 0],              // 2
  ['s', 0],              // 3
  ['t', 0],              // 4
  ['te', -1],            // 5
  ['st', -2],            // 6
  ['test', -3],          // 7
  ['tes', -4],           // 8
];

describe('BPE Tokenizer', () => {
  let tokenizer: bpe.Tokenizer;
  beforeAll(() => {
    tokenizer = new bpe.Tokenizer(vocab);
  });

  it('should tokenize correctly', () => {
    expect(tokenizer.encode('a test')).toEqual([0, 1, 0, 7]);
  });
});
