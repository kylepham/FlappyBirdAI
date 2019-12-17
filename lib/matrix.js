class Matrix
{
    constructor(rows, cols)
    {
        this.rows = rows;
        this.cols = cols;
        this.data = [];

        for (let i = 0; i < this.rows; i++)
        {
            this.data[i] = [];
            for (let j = 0; j < this.cols; j++)
                this.data[i][j] = 0;
        }
    }

    static fromArray(arr)
    {
        let m = new Matrix(arr.length, 1);
        for (let i = 0; i < m.rows; i++)
            for (let j = 0; j < m.cols; j++)
            m.data[i][j] = arr[i];
        return m;
    }

    toArray()
    {
        let arr = [];
        for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.cols; j++)
                arr.push(this.data[i][j]);
        return arr;
    }

    randomize()
    {
        for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.cols; j++)
                this.data[i][j] = Math.random() * 2 - 1;
    }

    add(n)
    {
        if (n instanceof Matrix)
        {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    this.data[i][j] += n.data[i][j];
        }
        else
        {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    this.data[i][j] += n;
        }
    }

    static subtract(a, b)
    {
        let result = new Matrix(a.rows, a.cols);
        for (let i = 0; i < result.rows; i++)
            for (let j = 0; j < result.cols; j++)
                result.data[i][j] = a.data[i][j] - b.data[i][j];
        return result;
    }

    static mult(a, b)
    {
        // console.log("IN STATIC MULT : ");
        // console.log("OUT STATIC MUL : ");
        //Matrix Multiplication
        if (a.cols !== b.rows)
        {
            console.log("Cols of A must match rows of B", b.rows, a);
            return;
        }
        let result = new Matrix(a.rows, b.cols);
        for (let i = 0; i < result.rows; i++)
        {
            for (let j = 0; j < result.cols; j++)
            {
                let sum = 0;
                for (let k = 0; k < a.cols; k++)
                {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    mult(a)
    {
        if (a instanceof Matrix)
        {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    this.data[i][j] *= a.data[i][j];
        }
        else
        {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    this.data[i][j] *= a;
        }
    }

    //A function acts as an argument to a method
    map(func)
    {
        for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.cols; j++)
                this.data[i][j] = func(this.data[i][j]);
    }

    static map(matrix, func)
    {
        let result = new Matrix(matrix.rows, matrix.cols);
        for (let i = 0; i < matrix.rows; i++)
            for (let j = 0; j < matrix.cols; j++)
                result.data[i][j] = func(matrix.data[i][j]);
        return result;
    }

    static transpose(matrix)
    {
        let result = new Matrix(matrix.cols, matrix.rows);
        for (let i = 0; i < matrix.rows; i++)
            for (let j = 0; j < matrix.cols; j++)
                result.data[j][i] = matrix.data[i][j];
        return result;
    }

    copy()
    {
        let result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.cols; j++)
            {
                result.data[i][j] = this.data[i][j];
            }
        return result;
    }

    print()
    {
        console.table(this.data);
    }
}
