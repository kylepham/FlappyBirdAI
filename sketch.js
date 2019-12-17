let screenWidth = 287; //Default width is 287
let screenHeight = 511; //Default width is 511;

let birds = [];
let totalBird = 200;
let pipes = [];
let endGame = false;
let passedPipe = -1;
let totalScore = 0;
let population = 0
let previousGeneration = [];
let count = 0;
let slider;
let imageAtlas, imageTopPipe, imageBottomPipe, imageBackground;
let imageBirds = [];
let generation = 0;
let allTimeHighScore = 0;

function preload()
{
    imageAtlas = loadImage("images/AtlasMain.png");
    imageBackground = loadImage("images/Background.png");
    imageTopPipe = loadImage("images/TopPipe.png");
    imageBottomPipe = loadImage("images/BottomPipe.png");
    imageBirds[0] = loadImage("images/Bird0.png");
    imageBirds[1] = loadImage("images/Bird1.png");
    imageBirds[2] = loadImage("images/Bird2.png");
}

function setup()
{
    tf.setBackend('cpu');
    createCanvas(screenWidth, screenHeight);
    for (let i = 0; i < totalBird; i++)
    {
        birds[i] = new Bird();
    }
    //pipes.push(new Pipe());
    population = createP();
    slider = createSlider(1, 200, 1);

}

function draw()
{
    // console.log(randomGaussian(0, 0.1));
    image(imageBackground, 0, 0);
    for (let i = 0; i < birds.length; i++)
        birds[i].show();
    for (let i = 0; i < pipes.length; i++)
        pipes[i].show();
    for (let n = 0; n < slider.value(); n++)
    {
        if (count % 75 === 0)
        {
            pipes.push(new Pipe());
            count = 0;
        }
        count++;
        for (let i = 0; i < birds.length; i++)
        {
            birds[i].decide(pipes);
            birds[i].update();
        }

        for (let i = 0; i < pipes.length; i++)
        {
            pipes[i].update();
            for (let j = 0; j < birds.length; j++)
            {
                if (pipes[i].x < -imageTopPipe.width)
                {
                    pipes.splice(i, 1);
                    passedPipe = -1;
                }
                if (pipes[i].x < birds[j].x && passedPipe !== i)
                {
                    passedPipe = i;
                    totalScore++;
                }

                if (pipes[i].collision(birds[j]) || birds[j].up_down_Collision())
                {
                    previousGeneration.push(birds[j]);
                    birds.splice(j, 1);
                }
            }
        }

        // Display score
        textAlign(CENTER);
        textSize(64);
        fill(0, 255, 0);
        stroke(0);
        strokeWeight(4);
        text(totalScore, screenWidth / 2, 100);

        //All birds are dead
        if (birds.length === 0)
        {
            pipes = [];
            count = 0;
            if (allTimeHighScore < totalScore)
            {
                allTimeHighScore = totalScore;
            }
            totalScore = 0;
            birdEvolution();
            previousGeneration = [];
            generation++;
        }
    }
    textAlign(LEFT);
    textSize(10);
    noStroke();
    fill(255, 200);
    text("Generation: " + generation, 10, 10);
    fill(255, 200);
    text("All time high score: " + allTimeHighScore, 10, 20);

    fill(127);
    rect(screenWidth, 0, width - screenWidth, height);
}

function birdEvolution()
{
    calculateFitness();
    for (let i = 0; i < totalBird; i++)
    {
        birds[i] = previousGene();
    }
    for (let i = 0; i < totalBird; i++)
    {
        previousGeneration[i].dispose();
    }
    previousGeneration = [];
}

function calculateFitness()
{
    let sum = 0;
    for (let i = 0; i < previousGeneration.length; i++)
    {
        sum += previousGeneration[i].lifeSpan;
    }

    for (let i = 0; i < previousGeneration.length; i++)
    {
        previousGeneration[i].fitness = previousGeneration[i].lifeSpan / sum;
    }
}

function previousGene()
{
    let percentage = random(1);
    let index = previousGeneration.length - 1;
    while (percentage > 0)
    {
        percentage -= previousGeneration[index].fitness;
        index--;
    }
    index++;
    let bird = previousGeneration[index];
    let child = new Bird(bird.brain);
    child.mutate();
    return child;
}
