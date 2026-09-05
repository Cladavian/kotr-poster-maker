# Putting the KOTR Poster Maker online (GitHub Pages)

The site is static: just these files. Nothing runs on a server, so hosting is free.

1. Create a GitHub account if you do not have one, then create a new **public** repository
   called `kotr-poster-maker` (leave it empty: no README, no .gitignore).
2. In this folder, run (replace YOURNAME with your GitHub username):

       git add -A
       git commit -m "KOTR poster maker"
       git remote add origin https://github.com/YOURNAME/kotr-poster-maker.git
       git push -u origin main

   Git will open a browser window to sign you in the first time.
3. On GitHub: repository → Settings → Pages → "Build and deployment" → Source: **Deploy from a branch**,
   Branch: **main**, folder **/ (root)** → Save.
4. After a minute the site is live at `https://YOURNAME.github.io/kotr-poster-maker/`.

Updating later: edit the files, then `git add -A`, `git commit -m "update"`, `git push`.

Notes
- The site ships with no posters. Everyone who uses it adds their own, and those stay in their browser
  (IndexedDB). The `posters/` folder is ignored by git for that reason.
- The AI model (93 MB) is downloaded by each visitor's browser from Hugging Face on first use and cached.
