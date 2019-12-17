class Bird
{
    constructor(brain)
    {
        this.y = screenHeight / 2;
        this.x = 64;
        this.gravity = 0.8;
        this.lift = -17;
        this.velocity = 0;
        this.lifeSpan = 0;
        this.fitness = 0;
        this.imageBirds = imageBirds;
        this.currentBirdImage = 0;
        this.spacebarTimer = 0;

        if (brain)
        {
            this.brain = brain.copy();
        }
        else
        {
            this.brain = new NeuralNetwork(4, 4, 2);
        }

    }

    dispose()
    {
        this.brain.dispose();
    }

    mutate()
    {
        this.brain.mutate(0.1);
    }

    show()
    {
        imageMode(CENTER);
        image(this.imageBirds[this.currentBirdImage], this.x, this.y);
    }

    update()
    {
        this.lifeSpan++;
        this.spacebarTimer++;
        if (this.lifeSpan % 10 === 3)
        {
            this.currentBirdImage = 0;
        }
        else if (this.lifeSpan % 10 === 6)
        {
            this.currentBirdImage = 1;
        }
        else if (this.lifeSpan % 10 === 9)
        {
            this.currentBirdImage = 2;
        }
        this.velocity += this.gravity;
        this.velocity *= 0.9;
        this.y += this.velocity;
    }

    frontPipe(pipes_list)
    {
        for (let i = 0; i < pipes_list.length; i++)
        {
            if (this.x < pipes_list[i].x + pipes_list[i].w)
            {
                return i;
            }
        }
    }

    decide(pipes_list)
    {
        if (!pipes_list.length)
        {
            return;
        }
        //Inputs array
        let inputs = [];

        //console.log(this.frontPipe(pipes_list));
        //The first element is the height of the bird
        inputs[0] = this.y / screenHeight;

        //The second element is the x_position of the pipes
        inputs[1] = pipes_list[this.frontPipe(pipes_list)].x / screenWidth;

        //The third element is the y_position of the above pipe
        inputs[2] = pipes_list[this.frontPipe(pipes_list)].top / screenHeight;

        //The third element is the y_position of the below pipe
        inputs[3] = pipes_list[this.frontPipe(pipes_list)].bottom / screenHeight;
        let outputs = this.brain.feedForward(inputs);
        if (outputs[0] > outputs[1])
        {
            this.up();
        }
    }

    up()
    {
        if (this.spacebarTimer > 6)
        {
            this.velocity += this.lift;
            this.velocity *= 0.9
            this.spacebarTimer = 0;
        }
        else return;
    }

    updateScore()
    {
        this.score++;
    }

    up_down_Collision()
    {
        if (this.y > screenHeight || this.y < 0)
            return true;
        return false;
    }
}
