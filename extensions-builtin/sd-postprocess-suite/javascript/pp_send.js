// Post Processing Suite - gallery extraction shim for the
// "Apply PP -> Send to img2img" button.
//
// Gradio runs this on the button's inputs BEFORE calling the python fn. The
// first input is the txt2img gallery; the rest are the post-processing UI
// values. We replace the gallery with just the *selected* image (mirroring
// Forge's built-in extract_image_from_gallery) and pass everything else
// through untouched, so the python handler receives:
//     (selected_image, *pp_settings)
function pp_extract_and_pass(...args) {
    const gallery = args[0];
    let img = null;

    if (gallery && gallery.length) {
        let index = -1;
        if (typeof selected_gallery_index === "function") {
            index = selected_gallery_index();
        }
        if (index < 0 || index >= gallery.length) {
            index = 0; // default to first image if nothing is selected
        }
        img = [gallery[index]];
    }

    return [img].concat(args.slice(1));
}
