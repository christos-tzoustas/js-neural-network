import weatherData from "./data/data-last-year.json" assert { type: "json" };
import {
  normalizeData,
  didItRain,
  mergeArrays,
  round2DecimalPoints,
} from "./utils/utils.js";
import { NeuralNetwork } from "./network.js";

// Normalize data
const {
  mean: tempMean,
  standardDeviation: tempStdDev,
  result: tempRes,
} = normalizeData(weatherData.hourly.temperature_2m);
const {
  mean: humMean,
  standardDeviation: humStdDev,
  result: humRes,
} = normalizeData(weatherData.hourly.relativehumidity_2m);
const normalizedData = {
  temperature: tempRes,
  humidity: humRes,
  rain: weatherData.hourly.rain.map(didItRain),
};

// Create inputs and targets
const inputsList = mergeArrays(
  normalizedData.temperature,
  normalizedData.humidity
);
const targetsList = normalizedData.rain.map((didItRain) => [didItRain]);

// Split to training and test data
const totalLengthDataPoints = normalizedData.temperature.length;
const trainingSetLength = Math.floor(0.8 * totalLengthDataPoints);
const trainInputs = inputsList.slice(0, trainingSetLength);
const trainTargets = targetsList.slice(0, trainingSetLength);
const testInputs = inputsList.slice(trainingSetLength);
const testTargets = targetsList.slice(trainingSetLength);


// Create network
const network = new NeuralNetwork(2, 2, 1);

// Train network
const epochs = 100;
const learningRate = 0.9;
network.train(trainInputs, trainTargets, epochs, learningRate);

// Test the network
let correct = 0;
for (let i = 0; i < testInputs.length; i++) {
  const prediction = network.predict(testInputs[i]);
  // Round the prediction to 0 or 1
  const roundedPrediction = Math.round(prediction[0]);

  // If the prediction matches the target, increment correct
  if (roundedPrediction === testTargets[i][0]) {
    correct++;
  }
}

// Calculate accuracy
const accuracyPercentage = (correct / testInputs.length) * 100;

// Logs
console.log("-------------");
console.log("Parameters");
console.log("-------------");
console.log(`Training set length: ${trainingSetLength}`);
console.log(`Test set length: ${totalLengthDataPoints - trainingSetLength}`);
console.log(`Epochs: ${epochs}`);
console.log(`Learning rate: ${learningRate}`);
console.log("-------------");
console.log("Results");
console.log("-------------");
console.log(`Accuracy: ${round2DecimalPoints(accuracyPercentage)}%`);

// Today's prediction
const todayTemp = 21;
const todayHum = 44;
const normalizedTemp = round2DecimalPoints((todayTemp - tempMean) / tempStdDev);
const normalizedHum = round2DecimalPoints((todayHum - humMean) / humStdDev);
const normalizedSetToday = [normalizedTemp, normalizedHum];
const prediction = network.predict(normalizedSetToday);
const roundedPrediction = Math.round(prediction[0]);
console.log("-------------");
console.log("Today");
console.log("-------------");
console.log(
  `Today's temp: ${todayTemp} / Today's temp normalized: ${normalizedSetToday[0]}`
);
console.log(
  `Today's hum: ${todayHum} / Today's hum normalized: ${normalizedSetToday[1]}`
);
console.log(`Today's prediction: ${roundedPrediction}`);
