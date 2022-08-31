/**
 * @fileoverview some useful exports to play around with the models &
 * tokenizers.
 *
 * Simple usage (see ./playground.html for more complete usage example):
 *
 * model = lit.Model('tiny');
 * model.load(progress => console.log('loading...', progress));
 * console.log(model.computeProbabilities(['a dog', 'a cat'], '0'));
 */

import {Model} from './lit_demo/compute';
import {getImageUrl, setBaseUrl} from './lit_demo/constants';
import {ImageData} from './lit_demo/data';
import * as tf from '@tensorflow/tfjs-core';

// tslint:disable-next-line:no-any Export symbols into global namespace.
(window as any).lit = { Model, getImageUrl, ImageData, setBaseUrl };
// tslint:disable-next-line:no-any Export symbols into global namespace.
// tslint:disable-next-line:ban-module-namespace-object-escape Export all of TF.
(window as any).tf = tf;
