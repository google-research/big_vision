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

import {Tokenizer} from './sentencepiece_unigram';

const stubbedTokenizerVocab = [
  ['�', 0],
  ['<s>', 0],
  ['</s>', 0],
  ['extra_token_id_1', 0],
  ['extra_token_id_2', 0],
  ['extra_token_id_3', 0],
  ['▁', -2],
  ['▁a', -1],
  ['▁ç', -2],
  ['a', -3],
  ['.', -1],
  ['▁I', -1],
  ['▁like', -1],
  ['▁it', -1],
  ['I', -2],
  ['like', -2],
  ['it', -2],
  ['l', -3],
  ['i', -3],
  ['k', -3],
  ['e', -3],
  ['i', -3],
  ['t', -3]
];

describe('Universal Sentence Encoder tokenizer', () => {
  let tokenizer: Tokenizer;
  beforeAll(() => {
    tokenizer = new Tokenizer(stubbedTokenizerVocab as Array<[string, number]>);
  });

  it('basic usage', () => {
    expect(tokenizer.encode('Ilikeit.')).toEqual([11, 15, 16, 10]);
  });

  it('handles whitespace', () => {
    expect(tokenizer.encode('I like it.')).toEqual([11, 12, 13, 10]);
  });

  it('should normalize inputs', () => {
    expect(tokenizer.encode('ça')).toEqual(tokenizer.encode('c\u0327a'));
  });

  it('should handle unknown inputs', () => {
    expect(() => tokenizer.encode('😹')).not.toThrow();
  });

  it('should treat consecutive unknown inputs as a single word', () => {
    expect(tokenizer.encode('a😹😹')).toEqual([7, 0]);
  });
});
