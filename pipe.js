class Pipe
{
    constructor()
    {
        this.top = random(40, 300);
        this.bottom = screenHeight - this.top - 150;
        this.x = screenWidth;
        this.w = imageTopPipe.width;
        this.speed = 3;
    }

    show()
    {
        imageMode(CORNER);
        image(imageTopPipe, this.x, this.top - imageTopPipe.height);
        image(imageBottomPipe, this.x, screenHeight - this.bottom);
    }

    update()
    {
        this.x -= this.speed;
    }

    collision(object)
    {
        if (object.y - imageBirds[0].height / 2 < this.top || object.y + imageBirds[0].height / 2 > screenHeight - this.bottom)
            if (object.x + imageBirds[0].width / 2 > this.x && object.x - imageBirds[0].width / 2 < this.x + this.w)
                return true;
        return false;
    }

}
