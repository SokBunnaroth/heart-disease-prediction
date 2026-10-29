
const HEART_DISEASE_MODEL = {
  numericalFeatures: ["age", "trestbps", "chol", "thalach", "oldpeak"],
  categoricalFeatures: [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
  ],

  numericalMedians: [56.0, 130.0, 244.5, 153.5, 0.8],

  scalerMean: [
    54.549586776859506,
    130.95867768595042,
    249.8388429752066,
    149.96280991735537,
    0.9991735537190083
  ],

  scalerScale: [
    8.978372765440131,
    17.58610302878636,
    52.737566192586975,
    22.639527461380734,
    1.1206178835595506
  ],

  categoricalModes: [1, 4, 0, 0, 0, 1, 0, 3],

  categories: [
    [0, 1],
    [1, 2, 3, 4],
    [0, 1],
    [0, 1, 2],
    [0, 1],
    [1, 2, 3],
    [0, 1, 2, 3],
    [3, 6, 7]
  ],

  coefficients: [
    -0.060936179353880376,
    0.340559183576034,
    0.17891988640737652,
    -0.3724738712062912,
    0.25952539845836015,
    -0.70083589088973,
    0.6903530973764144,
    -0.6191759931524464,
    0.15289842147342297,
    -0.5747658356901358,
    1.030560613855844,
    0.09285943309201511,
    -0.10334222660533039,
    -0.15577500264448363,
    -0.06773008596138341,
    0.21302229509255174,
    -0.27011479216515505,
    0.2596319986518401,
    -0.44063939944452724,
    0.6034274153709913,
    -0.17327080943977893,
    -1.4986407773210317,
    0.35175424577590825,
    0.6421710867063076,
    0.4942326513255008,
    -0.5402428950844352,
    -0.28873061185026566,
    0.8184907134213866
  ],

  intercept: -0.02579041651056184
};

function predictHeartDisease(input) {
  const model = HEART_DISEASE_MODEL;
  const transformed = [];

  // 1. Impute missing numerical values and standardize them.
  model.numericalFeatures.forEach((feature, i) => {
    let value = input[feature];

    if (value === undefined || value === null || value === "") {
      value = model.numericalMedians[i];
    }

    value = Number(value);

    if (!Number.isFinite(value)) {
      throw new Error(`Invalid numerical value for ${feature}.`);
    }

    transformed.push(
      (value - model.scalerMean[i]) / model.scalerScale[i]
    );
  });

  // 2. Impute missing categorical values and one-hot encode them.
  model.categoricalFeatures.forEach((feature, i) => {
    let value = input[feature];

    if (value === undefined || value === null || value === "") {
      value = model.categoricalModes[i];
    }

    value = Number(value);

    if (!Number.isFinite(value)) {
      throw new Error(`Invalid categorical value for ${feature}.`);
    }

    const knownCategories = model.categories[i];

    // OneHotEncoder(handle_unknown="ignore") produces all zeros
    // for a category that was not present during training.
    knownCategories.forEach(category => {
      transformed.push(value === category ? 1 : 0);
    });
  });

  if (transformed.length !== model.coefficients.length) {
    throw new Error("The transformed feature count does not match the model.");
  }

  // 3. Calculate Logistic Regression probability.
  let score = model.intercept;

  transformed.forEach((value, i) => {
    score += value * model.coefficients[i];
  });

  // Numerically stable sigmoid.
  let probability;
  if (score >= 0) {
    probability = 1 / (1 + Math.exp(-score));
  } else {
    const expScore = Math.exp(score);
    probability = expScore / (1 + expScore);
  }

  return {
    prediction: probability >= 0.5 ? 1 : 0,
    probability: probability,
    percentage: Number((probability * 100).toFixed(1))
  };
}
