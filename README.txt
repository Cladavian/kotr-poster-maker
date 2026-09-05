KOTR POSTER MAKER
=================
Turns real fight posters (UFC, PFL, Rizin, AEW, boxing...) into KOTR posters:
the original fighters and text are erased by an AI eraser running in your browser,
then your fighters, names, logo and sponsors go on top.

Open index.html in Chrome or Edge, or use the hosted version (see DEPLOY.md).
Everything runs in the browser. Photos and posters never leave your PC.

Workflow
--------
1. POSTER LIBRARY
   Save real posters you like into the "posters" folder, then "Add folder" (or "Add posters").
   They are stored in the browser, so you only do this once. Click one to use it, or press
   "Best fit for these fighters" to rank the library for your two fighters.

2. CLEAN THE POSTER  (once per poster)
   Press "Clean this poster". In the cleaner:
   - "Auto-mark everything" marks the original fighters, all text and text logos
     (any font, any direction) in one go. Takes about a minute the first time.
   - Anything left over: "Wand" and click a letter or logo = everything in that
     colour gets marked (perfect for the giant side names). "Rectangle", "Brush",
     "Un-mark", and the Top / Bottom / Sides buttons for the usual text bands.
   - "Erase & rebuild" runs the AI eraser. First use downloads the model once (93 MB),
     then roughly 1-3 minutes per poster on CPU with "High quality" on, a few seconds
     with a supported GPU. The page is busy while it runs; that is normal.
   - Check the result. "Refine further" lets you mark leftovers and run again.
     "Keep result" saves the cleaned poster in your library (marked CLEANED).
   Tip: erase a little more than you think. A margin around fighters and text
   avoids ghost outlines.

3. FIGHTERS
   Type name, nickname, record. Upload the photo: transparent PNGs drop straight in.
   For a plain screenshot, click its background colour and press "Apply colour cut-out",
   or use "AI cut-out". Use Size / Move / Mirror so the fighters face each other and
   cover the area where the originals stood.

4. LOGO, SPONSORS, EXPORT
   Branding & Style: replace the logo, pick colours. Sponsors: add logos for the bottom strip.
   Text uses the poster's own accent colour by default.
   "Download PNG (2x quality)" or "Copy to clipboard" and paste into Discord.

Other
-----
- "KOTR design" mode is the built-in black-and-gold layout, no source poster needed.
- "Save project (.json)" keeps photos + text so you can reload later.
- Fonts (Anton, Bebas Neue, Oswald) and the AI models load from the internet on first use.
