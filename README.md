# Aegis image stylization node

Powered by models from [Arbitrary Image Stylization TFJS](https://github.com/reiinakano/arbitrary-image-stylization-tfjs) by reiinakano (Reiichiro Nakano).

## Environment
* `PORT` - the port to listen on
* `STYLE_NAME` - name of the style model to use; options are `mobilenet-style` (default) or `inception-style`
* `TRANSFORMER_NAME` - name of the transformer model to use; options are `seperable-transformer` (default) or `original-transformer`
* `SIZE` - maximum dimension of the image to generate (defaults to 512)
* `STYLE_SCALE` - style scale to use if not overridden by request (defaults to 0.5)
* `STYLE_RATIO` - strength of style to use if not overridden by request (defaults to 0.8)

## Usage
POST the following data to `/stylize`:
```js
{
  content: "<base64 encoded content image>",
  styles: ["<base64 encoded style image>" ...],
  scale: 0.5, //optional; any real number 0.25..2 suggested
  ratio: 0.8 //optional; 0..1 range
}
```

## TODO
* Create routes for generating/combining style vectors and stylizing with previously created style vectors
