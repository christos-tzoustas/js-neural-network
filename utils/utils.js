export const calculateMean = (listNumbers) => {
  const sum = add(listNumbers);
  const average = sum / listNumbers.length;

  return average;
};

const add = (listNumbers) => {
  if (listNumbers.length === 0) {
    return 0;
  }

  return listNumbers[0] + add(listNumbers.slice(1));
};

const square = (x) => x * x;

export const calculateStandardDeviation = (listNumbers) => {
  const mean1 = calculateMean(listNumbers);
  const meanSubtractedAndSquared = listNumbers.map((num) =>
    square(num - mean1)
  );
  const mean2 = calculateMean(meanSubtractedAndSquared);
  const standardDeviation = sqrt(mean2);

  return round2DecimalPoints(standardDeviation);
};

export const round2DecimalPoints = (num) => Math.round(num * 100) / 100;

const sqrt = (x) => {
  const iter = (x, guess) => {
    if (isGoodEnough(x, guess)) {
      return guess;
    } else {
      return iter(x, improveGuess(x, guess));
    }
  };

  return iter(x, 1);
};

const improveGuess = (x, guess) => {
  const improve = (y) => x / y;

  return averageDamp(improve)(guess);
};

const averageDamp = (f) => (x) => {
  return (f(x) + x) / 2;
};

const isGoodEnough = (x, guess) => {
  const tolerance = 0.00001;
  return Math.abs(x - square(guess)) < tolerance;
};

export const normalizeData = (list) => {
  const mean = calculateMean(list);

  const standardDeviation = calculateStandardDeviation(list);

  const result = list.map((val) =>
    round2DecimalPoints((val - mean) / standardDeviation)
  );

  return { result, mean, standardDeviation };
};

export const didItRain = (mm) => (mm > 0 ? 1 : 0);

export const mergeArrays = (array1, array2) => {
  if (array1.length !== array2.length) {
    throw new Error("Arrays are not of the same length");
  }
  return array1.map((item, index) => [item, array2[index]]);
};
