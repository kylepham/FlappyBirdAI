function sigmoid(x)
{
    return 1 / (1 + Math.exp(-5 * x));
}

function dsigmoid(x)
{
    return x * (1 - x);
}

class NeuralNetwork
{
    constructor(inp, hid, out)
    {
        if (inp instanceof NeuralNetwork)
        {
            this.inp = inp.inp;
            this.hid = inp.hid;
            this.out = inp.out;

            this.weights_ih = inp.weights_ih.copy();
            this.weights_ho = inp.weights_ho.copy();

            this.bias_h = inp.bias_h.copy();
            this.bias_o = inp.bias_o.copy();
        }
        else
        {
            this.inp = inp;
            this.hid = hid;
            this.out = out;
            this.weights_ih = new Matrix(this.hid, this.inp);
            this.weights_ho = new Matrix(this.out, this.hid);

            this.weights_ih.randomize();
            this.weights_ho.randomize();

            this.bias_h = new Matrix(this.hid, 1);
            this.bias_h.randomize();
            this.bias_o = new Matrix(this.out, 1);
            this.bias_o.randomize();

        }

        this.learning_rate = 0.1;
    }

    feedForward(inputs_array)
    {
        //console.log(inputs);
        //Generating hidden output
        let inputs = Matrix.fromArray(inputs_array);
        let hidden = Matrix.mult(this.weights_ih, inputs);
        hidden.add(this.bias_h);
        //activation function
        hidden.map(sigmoid);

        let outputs = Matrix.mult(this.weights_ho, hidden);
        outputs.add(this.bias_o);
        outputs.map(sigmoid);

        return outputs.toArray();
    }

    train(inputs_array, targets_array)
    {
        //console.log("trainarr: " + inputs_array + " traintar: " + targets_array);
        //Generating hidden output
        let inputs = Matrix.fromArray(inputs_array);
        let hidden = Matrix.mult(this.weights_ih, inputs);

        hidden.add(this.bias_h);
        //activation function
        hidden.map(sigmoid);

        let outputs = Matrix.mult(this.weights_ho, hidden);
        outputs.add(this.bias_o);
        outputs.map(sigmoid);

        //Convert array to Matrix Object
        let targets = Matrix.fromArray(targets_array);

        // ERROR = TARGETS - OUTPUTS
        let outputs_error = Matrix.subtract(targets, outputs);

        let gradient = Matrix.map(outputs, dsigmoid);
        gradient.mult(outputs_error);
        gradient.mult(this.learning_rate);

        let hidden_T = Matrix.transpose(hidden);
        let weight_ho_deltas = Matrix.mult(gradient, hidden_T);

        this.weights_ho.add(weight_ho_deltas);
        this.bias_o.add(gradient);

        //Calculate the hidden layers errors
        let weights_ho_transposed = Matrix.transpose(this.weights_ho);
        let hidden_errors = Matrix.mult(weights_ho_transposed, outputs_error);

        //Calculate hidden gradient
        let hidden_gradient = Matrix.map(hidden, dsigmoid);
        hidden_gradient.mult(hidden_errors);
        hidden_gradient.mult(this.learning_rate);

        //Calculate input-hidden deltas
        let inputs_T = Matrix.transpose(inputs);
        let weights_ih_deltas = Matrix.mult(hidden_gradient, inputs_T);

        this.weights_ih.add(weights_ih_deltas);
        this.bias_h.add(hidden_gradient);
    }

    copy()
    {
        return new NeuralNetwork(this);
    }

    mutate(rate)
    {
        function mutate(val)
        {
            if (Math.random() < rate)
            {
                return val + randomGaussian(0, 0.1);
            }
            else
            {
                return val;
            }
        }

        this.weights_ih.map(mutate);
        this.weights_ho.map(mutate);
        this.bias_h.map(mutate);
        this.bias_o.map(mutate);
    }

    crossOver(partner, rate)
    {
        for (let i = 0; i < this.weights_ih.rows; i++)
            for (let j = 0; j < this.weights_ih.cols; j++)
            {
                if (random(1) < rate)
                {
                    this.weights_ih.data[i][j] = partner.weights_ih.data[i][j];
                }
            }
        for (let i = 0; i < this.weights_ho.rows; i++)
            for (let j = 0; j < this.weights_ho.cols; j++)
            {
                if (random(1) < rate)
                {
                    this.weights_ho.data[i][j] = partner.weights_ho.data[i][j];
                }
            }
        for (let i = 0; i < this.bias_h.rows; i++)
            for (let j = 0; j < this.bias_h.cols; j++)
            {
                if (random(1) < rate)
                {
                        this.bias_h.data[i][j] = partner.bias_h.data[i][j];
                }
            }
        for (let i = 0; i < this.bias_o.rows; i++)
            for (let j = 0; j < this.bias_o.cols; j++)
            {
                if (random(1) < rate)
                {
                    this.bias_o.data[i][j] = partner.bias_o.data[i][j];
                }
            }
    }

    //Print all weights
    print()
    {
        console.table(this.weights_ih.data);
        console.table(this.weights_ho.data);
        console.table(this.bias_h.data);
        console.table(this.bias_o.data);
    }
}
