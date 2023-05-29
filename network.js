export class Neuron {
  constructor(numInputs) {
    this.weights = new Array(numInputs).fill(0).map(() => Math.random());
    this.bias = Math.random();
  }

  calculateOutput(inputs) {
    let sum = this.bias;
    for (let i = 0; i < this.weights.length; i++) {
      sum += this.weights[i] * inputs[i];
    }
    this.output = this.sigmoid(sum);
    return this.output;
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
}

export class Layer {
  constructor(numNeurons, numInputsPerNeuron) {
    this.neurons = new Array(numNeurons)
      .fill(null)
      .map(() => new Neuron(numInputsPerNeuron));
  }

  feedForward(inputs) {
    const outputs = [];
    for (let neuron of this.neurons) {
      outputs.push(neuron.calculateOutput(inputs));
    }
    return outputs;
  }
}

export class NeuralNetwork {
  constructor(numInputs, numHidden, numOutputs) {
    this.inputLayerSize = numInputs;
    this.hiddenLayer = new Layer(numHidden, numInputs);
    this.outputLayer = new Layer(numOutputs, numHidden);
  }

  predict(inputs) {
    const hiddenOutputs = this.hiddenLayer.feedForward(inputs);
    const output = this.outputLayer.feedForward(hiddenOutputs);
    return output;
  }

  calculateLoss(actualOutputs, predictedOutputs) {
    let totalLoss = 0;
    for (let i = 0; i < actualOutputs.length; i++) {
      totalLoss -=
        actualOutputs[i] * Math.log(predictedOutputs[i]) +
        (1 - actualOutputs[i]) * Math.log(1 - predictedOutputs[i]);
    }
    return totalLoss / actualOutputs.length; // Average loss
  }

  train(inputsList, targetsList, epochs, learningRate) {
    if (inputsList.length !== targetsList.length) {
      throw new Error("inputsList and targetsList must have the same length");
    }

    for (let e = 0; e < epochs; e++) {
      // Reset updates
      let weightUpdates = Array(this.outputLayer.neurons.length)
        .fill(0)
        .map(() => Array(this.hiddenLayer.neurons.length).fill(0));
      let biasUpdates = Array(this.outputLayer.neurons.length).fill(0);
      let hiddenWeightUpdates = Array(this.hiddenLayer.neurons.length)
        .fill(0)
        .map(() => Array(this.inputLayerSize).fill(0));
      let hiddenBiasUpdates = Array(this.hiddenLayer.neurons.length).fill(0);

      // Accumulate updates
      inputsList.forEach((inputs, i) => {
        const targets = targetsList[i];

        const output = this.predict(inputs);

        // Compute the derivative of the loss with respect to output
        const outputError = output.map((o, j) => -(targets[j] - o));

        // Calculate updates
        for (let j = 0; j < this.outputLayer.neurons.length; j++) {
          const neuron = this.outputLayer.neurons[j];
          for (let k = 0; k < neuron.weights.length; k++) {
            let weightUpdate =
              learningRate *
              outputError[j] *
              neuron.output *
              (1 - neuron.output) *
              this.hiddenLayer.neurons[k].output;

            weightUpdates[j][k] += weightUpdate;
          }
          let biasUpdate =
            learningRate * outputError[j] * neuron.output * (1 - neuron.output);

          biasUpdates[j] += biasUpdate;
        }

        // Propagate the error back to the hidden layer
        const hiddenErrors = this.hiddenLayer.neurons.map((hiddenNeuron, i) => {
          return this.outputLayer.neurons.reduce(
            (sum, outputNeuron, k) =>
              sum + outputNeuron.weights[i] * outputError[k],
            0
          );
        });

        if (hiddenErrors.some(isNaN))
          // Calculate updates for hidden layer
          for (let j = 0; j < this.hiddenLayer.neurons.length; j++) {
            const neuron = this.hiddenLayer.neurons[j];
            for (let k = 0; k < neuron.weights.length; k++) {
              let hiddenWeightUpdate =
                learningRate *
                hiddenErrors[j] *
                neuron.output *
                (1 - neuron.output) *
                inputs[k];
              if (isNaN(hiddenWeightUpdate))
                hiddenWeightUpdates[j][k] += hiddenWeightUpdate;
            }
            let hiddenBiasUpdate =
              learningRate *
              hiddenErrors[j] *
              neuron.output *
              (1 - neuron.output);

            hiddenBiasUpdates[j] += hiddenBiasUpdate;
          }
      });

      // Calculate average updates and apply them
      let numExamples = inputsList.length;
      for (let j = 0; j < this.outputLayer.neurons.length; j++) {
        const neuron = this.outputLayer.neurons[j];
        for (let k = 0; k < neuron.weights.length; k++) {
          neuron.weights[k] -= weightUpdates[j][k] / numExamples;
        }
        neuron.bias -= biasUpdates[j] / numExamples;
      }
      for (let j = 0; j < this.hiddenLayer.neurons.length; j++) {
        const neuron = this.hiddenLayer.neurons[j];
        for (let k = 0; k < neuron.weights.length; k++) {
          neuron.weights[k] -= hiddenWeightUpdates[j][k] / numExamples;
        }
        neuron.bias -= hiddenBiasUpdates[j] / numExamples;
      }
    }
  }
}
