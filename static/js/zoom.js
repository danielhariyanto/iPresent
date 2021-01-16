let listImages

async function getImages() {
    try {
        const random = Math.ceil((Math.random() * 20))
        const result = await fetch(`https://randomuser.me/portraits/men/${random}.jpg`, {mode: 'cors'})
        const img = await result.json();
        listImages.push(img)
        return img;
    } catch (error) {
        console.log(error);
    }
} 

for (let i = 0; i < 6; i++) {
    getImages()
}

console.log(listImages);
//document.querySelector('.zoom-grid')
