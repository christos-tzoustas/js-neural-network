import { NeuralNetwork } from "../network.js";

const inputsList = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const targetsList = [[0], [1], [1], [0]];

const network = new NeuralNetwork(2, 8, 1);
network.train(inputsList, targetsList, 2000, 0.6);

console.log(network.predict([0, 0])); // should be close to 0
console.log(network.predict([0, 1])); // should be close to 1
console.log(network.predict([1, 0])); // should be close to 1
console.log(network.predict([1, 1])); // should be close to 0
