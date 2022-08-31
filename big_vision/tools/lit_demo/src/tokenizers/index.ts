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
