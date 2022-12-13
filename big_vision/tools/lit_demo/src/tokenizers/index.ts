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
 * @fileoverview Tokenizers and tokenizer mappings.
 */

import {Tokenizer, TokenizerConstructor, Vocabulary} from './common';
import * as sentencepieceBpe from './sentencepiece_bpe';
import * as sentencepieceUnigram from './sentencepiece_unigram';

export {Tokenizer, Vocabulary} from './common';

const TOKENIZERS = new Map<string, TokenizerConstructor>([
  ['BPE', sentencepieceBpe.Tokenizer],
  ['UNIGRAM', sentencepieceUnigram.Tokenizer],
]);

/**
 * Returns a tokenizer of type `name` using `vocabulary`.
 */
export const getTokenizer = (name: string, vocabulary: Vocabulary): Tokenizer => {
  const ctor = TOKENIZERS.get(name);
  if (!ctor) throw new Error(`Unknown tokenizer: ${name}`);
  return new ctor(vocabulary);
};
