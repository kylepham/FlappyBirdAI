class NeuralNetwork
{
    constructor(a, b, c, d)
    {
        if (a instanceof tf.Sequential)
        {
            this.model = a;
            this.inputs = b;
            this.hiddens = c;
            this.outputs = d;
        }
        else
        {
            this.inputs = a;
            this.hiddens = b;
            this.outputs = c;
            this.model = this.createModel();
        }
    }

    dispose()
    {
        this.model.dispose();
    }

    createModel()
    {
        const model = tf.sequential();
        const hidden = tf.layers.dense({
            units: this.hiddens,
            inputDim: this.inputs, // shape of inputs
            activation: 'sigmoid' // activation function
        });
        model.add(hidden);
        const output = tf.layers.dense({
            units: this.outputs,
            activation: 'softmax'
        })
        model.add(output); 
        return model;   
    }

    feedForward(inputs)
    {
        return tf.tidy(() => {
            const xs = tf.tensor2d([inputs]);
            const ys = this.model.predict(xs);
            const outputs = ys.dataSync();
            return outputs;
        });
    }

    copy()
    {
        return tf.tidy(() => {
            const modelCopy = this.createModel();
            const weights = this.model.getWeights(); // get weights
            const weightCopies = [];
            for (let i = 0; i < weights.length; i++)
                weightCopies[i] = weights[i].clone();
            modelCopy.setWeights(weightCopies); // and set weights
            return new NeuralNetwork(modelCopy, this.inputs, this.hiddens, this.outputs);
        });
    }

    mutate(rate)
    {
        tf.tidy(() => {
            const weights = this.model.getWeights();
            const mutatedWeights = [];
            for (let i = 0; i < weights.length; i++)
            {
                let tensor = weights[i];
                let shape = weights[i].shape;
                let values = tensor.dataSync().slice();
                for (let j = 0; j < values.length; j++)
                {
                    if (random(1) < rate)
                        values[j] += randomGaussian();
                }
                let newTensor = tf.tensor(values, shape);
                mutatedWeights[i] = newTensor;
            }
            this.model.setWeights(mutatedWeights);
        });
    }
}