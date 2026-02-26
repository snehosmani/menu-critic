# Menu Critic (Streamlit + Groq)

Menu Critic is a Streamlit multipage app that critiques restaurant menus to improve conversion, average order value (AOV), and customer experience.

## Live Demo

- Streamlit app: https://menu-critic-square-demo.streamlit.app/Menu_Critic

It supports:

- Pasted menu text (English)
- Uploaded menu image (JPG/PNG) with Groq vision extraction
- Two modes:
  - `Fix my menu` (serious, revenue-focused)
  - `Roast my menu` (funny but not mean)

## Project Structure

- `app.py` - landing/home page for the multipage app
- `pages/01_Menu_Critic.py` - main analysis workflow
- `pages/02_About.py` - product overview + how it works
- `pages/03_Why_I_Built_This.py` - short Square/Block-style pitch page
- `menu_critic_core.py` - Groq calls, image preprocessing, JSON validation, constants
- `sample_menu.txt` - demo input text
- `assets/` - local GIF placeholders for error/fallback UI states

## Local Run

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create local Streamlit secrets file at `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

4. Run the app:

```bash
streamlit run app.py
```

If the key is missing, the app shows a setup message and stops on the analysis page instead of crashing.

## Streamlit Community Cloud Deployment

1. Push this repo to GitHub.
2. Create a new app in Streamlit Community Cloud and select this repo.
3. Set the main file to `app.py`.
4. Add secrets in **Settings -> Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

5. Deploy.

### Hosted app (current)

- https://menu-critic-square-demo.streamlit.app/Menu_Critic

## GIF Assets (Local Files)

The app references local GIFs and does not download them automatically. Add these files to `assets/`:

- `assets/sad.gif` (API/rate-limit/general request failures)
- `assets/confused.gif` (image parsing / extraction failure)
- `assets/this_is_fine.gif` (invalid JSON response fallback)

If they are missing, the app shows a text placeholder instead.

## Free-tier / Rate-limit Notes

- Groq free-tier quotas can cause temporary `429` errors.
- The app includes a friendly error state ("Groq is taking a nap...") for this.
- A session rate limit of 1 request every 10 seconds is enforced to reduce accidental spam.

## Model Configuration

Default model choices are defined at the top of `menu_critic_core.py`:

- `TEXT_MODEL`
- `VISION_MODEL`

The app first attempts strict JSON schema output and falls back to `json_object` mode if needed for compatibility.

## Demo Input

Use `sample_menu.txt` to test the app quickly.
