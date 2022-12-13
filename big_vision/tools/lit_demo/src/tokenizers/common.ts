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
 * @fileoverview Utility code shared between tokenizers.
 */

/**
 * A vocabulary consists of a list of tokens, and optional numerical value.
 * The numerical value is used by the unigram algorithnm to find the best
 * tokenizaion, and is ignored by the BPE algorithm.
 */
export type Vocabulary = Array<[string, number]>;

/**
 * Converts a string to a sequence of tokens.
 */
export interface Tokenizer {
  encode(input: string): number[];
}

/**
 * Factory for new `Tokenizer`.
 */
export interface TokenizerConstructor {
  new (vocabulary: Vocabulary): Tokenizer;
}

/**
 * Unicode-aware character iteration of strings.
 */
export const stringToChars = (input: string): string[] => {
  const symbols = [];
  for (const symbol of input) {
    symbols.push(symbol);
  }
  return symbols;
};

/**
 * Special separator character used to delimit sub-word tokens.
 */
export const TOKEN_SEPARATOR =
  '\u2581';  // This is the unicode character 'lower one eighth block'.
