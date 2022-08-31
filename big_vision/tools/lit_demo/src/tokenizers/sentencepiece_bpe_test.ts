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
