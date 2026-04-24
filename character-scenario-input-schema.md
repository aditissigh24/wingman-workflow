# Character & Scenario Creation — Input → DB Field Derivation Map

> This document defines the exact user inputs required to generate all database fields
> for the `Character`, `Scenario`, and `ScenarioBeat` tables via LLM prompt.
> The goal: minimum creative input from a human → maximum dramatic, culturally grounded output from AI.

---

## PART 1 — CHARACTER CREATION

### User Inputs (6 fields)

---

#### INPUT C1 — Archetype Phrase
```
Label:   "Who is she in one punchy line?"
Type:    Short text (max 8 words)
Example: "The Girl Next Door Who Got Hot"
Example: "Bestfriend Ki Behen"
Example: "Rich Girl Slumming It On Bumble"
```

**Why it's required:**
This is the creative seed. Every other field traces back to this line.
It defines her social position, her irony, and the dramatic tension she carries.

---

#### INPUT C2 — Core Life Tension
```
Label:   "What is the one unresolved thing in her life right now?"
Type:    1–2 sentences, free text
Example: "30+ DMs daily, tired of creeps, but still secretly hoping
          someone real shows up"
Example: "Rohit ka best friend hai tu — aur woh separately text
          karne lagi hai. Koi nahi jaanta."
```

**Why it's required:**
Not her backstory. Just the **conflict** that makes her interesting to talk to.
This is what separates a character from a profile. Without tension there is no story.

---

#### INPUT C3 — City
```
Label:   "Which city is she from?"
Type:    Dropdown
Options: Indore / Lucknow / Mumbai / Delhi / Pune / Jaipur /
         Chandigarh / Hyderabad / Bengaluru / Kolkata
```

**Why it's required:**
City is not just a location. It is a dialect, a cultural reference pool,
a class signal, and a social context all at once.
- Indore → Hinglish, "bhai" as warmth, poha, middle-class glow-up
- Lucknow → Hindi-heavy, tehzeeb, homely, familiar warmth
- Mumbai → Code-switching, fast, independent, aspirational
- Delhi → Confident, slightly aggressive, brand-aware

---

#### INPUT C4 — Signature Communication Behavior
```
Label:   "What is one specific thing about how she texts or talks
          that makes her feel real?"
Type:    1 sentence, free text
Example: "Deliberately waits 5–10 minutes before replying even
          though she typed it instantly"
Example: "Sends voice notes only when she actually trusts someone"
Example: "Uses your name more than normal when she's nervous —
          it feels intimate without meaning to"
```

**Why it's required:**
Personality adjectives are generic. One behavioral tell is cinematic.
This single input generates the entire communication layer of the character.

---

#### INPUT C5 — What She Never Does
```
Label:   "What is one thing she never does, no matter what?"
Type:    1 sentence, free text
Example: "Never says she likes someone first — shows it through
          behavior and small observations instead"
Example: "Never pretends the situation isn't complicated —
          the weight of it is what makes it real"
Example: "Never double texts first. Ever."
```

**Why it's required:**
This is the creative constraint that makes the character feel consistent
across hundreds of user conversations. It is the soul of `hardLimits`.

---

#### INPUT C6 — Physical Vibe (5 words)
```
Label:   "Describe how she looks and feels in 5 words"
Type:    Short text
Example: "Fit, casual, golden hour, homely"
Example: "Churidar, mischievous smile, familiar warmth"
Example: "Oversized tee, messy bun, tired eyes, real"
```

**Why it's required:**
Five adjectives give the image generation model enough to be specific
without over-prescribing. More than 5 words dilutes the visual coherence.

---

### DB Field Derivation Map — Character

| DB Field | Source Inputs | Derivation Logic |
|---|---|---|
| `name` | C1, C3 | AI picks a real Indian first name that fits the archetype and city |
| `age` | C2 | AI infers age from the tension (college = 19–21, working = 22–26) |
| `city` | C3 | Direct |
| `gender` | — | Defaults to FEMALE; add a toggle if needed later |
| `archetype` | C1 | Direct |
| `vibeSummary` | C1, C2 | AI writes 2–3 lines in Hinglish: who she is + why she's worth talking to + the hook |
| `backstory` | C2, C3 | AI expands the tension into a 4–5 line culturally specific narrative |
| `speakingStyle` | C3, C4 | AI derives dialect pattern, filler words, reply length behavior from city + behavior |
| `emojiUsage` | C4, C5 | AI infers from behavioral tells (e.g., waits to reply = deliberate, controlled emoji use) |
| `textingSpeed` | C4 | Direct behavioral translation |
| `voicePrompt` | C3, C4, C5 | Full LLM character card: how to stay in voice, what to do when nervous/comfortable/irritated |
| `hardLimits` | C5 | AI generates 4–5 rules as character truths, not restrictions |
| `avatarPrompt` | C3, C6 | AI builds image gen prompt: person + vibe + city-appropriate setting + lighting |
| `accentHsl` | C1, C6 | AI maps archetype + physical vibe to a color palette (warm/cool/earthy) |
| `avatarImage` | avatarPrompt | Image generation pipeline (ElevenLabs / Midjourney / Flux) |
| `imageUrl` | avatarPrompt | Same pipeline, stored URL |
| `voiceAudio` | voicePrompt + C3 | TTS pipeline (ElevenLabs) with accent param from city |

---

## PART 2 — SCENARIO CREATION

### User Inputs (5 fields)

---

#### INPUT S1 — The Specific Trigger Detail
```
Label:   "What exactly is happening right now that puts the user
          in this situation? Be hyper-specific."
Type:    1–2 sentences, free text
Example: "She posted an Indori poha story. Tu 48th DM hai uske inbox mein."
Example: "She swiped right on Bumble. Teri profile mein worst
          lighting wali photo hai aur bio mein sirf 'chai > coffee' hai."
Example: "Tu uske ghar aaya tha Rohit se milne. Rohit bahar gaya.
          Woh akeli hai. Chai pooch rahi hai."
```

**Why it's required:**
This is the most important scenario input. The specificity of the trigger
is what makes the `situationSetupForUser` feel immersive rather than generic.
"They matched on Bumble" produces nothing. "Her bio says LSE '23 and her
first photo is from a business class seat" produces everything.

---

#### INPUT S2 — The User's Primal Fear
```
Label:   "What fear or insecurity does this situation poke at
          in the person playing it?"
Type:    1–2 sentences, free text
Example: "You're one of 47. You're invisible unless you do
          something actually different."
Example: "Her world is bigger than yours. Do you shrink
          or do you stand?"
Example: "You've been friendzoned here your whole life.
          Tonight something shifted. Do you say it or
          let it pass again?"
```

**Why it's required:**
This is what makes users come back to replay the scenario.
It is the emotional engine of the entire experience.
Without a clear primal fear, the scenario is just a chat — not a challenge.

---

#### INPUT S3 — Her Emotional State Right Now
```
Label:   "How is she feeling at the exact moment this scenario starts?
          Not in general — right now."
Type:    1–2 sentences, free text
Example: "Mildly curious but fundamentally skeptical.
          She has been disappointed too many times."
Example: "Bored by impressive men for years. This match
          feels slightly different — she is paying attention
          to find out if that holds."
Example: "Nervous but pretending not to be. She made the
          move and now she has to live with it."
```

**Why it's required:**
This is Beat 1's emotional state and the entire scenario's opening tone.
It determines whether she's cold-to-warm or warm-to-complicated.
The trajectory of all beats derives from this starting point.

---

#### INPUT S4 — Time and Place
```
Label:   "Where is she physically, and what time is it?"
Type:    2 short fields or one combined line
Example: "Late night, her bedroom, post-gym, phone in hand"
Example: "Afternoon, probably a nice café, just opened the app"
Example: "Evening, her home, Rohit just stepped out, she's
          in the kitchen"
```

**Why it's required:**
Time of day changes the emotional register completely.
Late night = intimate, lower guards. Afternoon = casual, less charged.
Evening = transition energy, things can go either way.
Place gives the image generation model a concrete visual anchor.

---

#### INPUT S5 — Arc Destination
```
Label:   "Where does this scenario end if the user does
          everything right? One sentence."
Type:    1 sentence, free text
Example: "From one of 47 strangers to the one she
          actually wants to talk to"
Example: "The class gap disappears — she stops thinking
          about his salary and starts thinking about him"
Example: "He finally says it. She already knew.
          Now it's real."
```

**Why it's required:**
This gives the LLM the narrative destination. Without it, beats drift.
With it, every beat is written in service of this arc, and the story
feels intentional rather than procedural.

---

### DB Field Derivation Map — Scenario

| DB Field | Source Inputs | Derivation Logic |
|---|---|---|
| `scenarioTitle` | S1, S2 | AI writes a Hinglish title (max 8 words) that names the tension, not just the situation |
| `tagline` | S2, S5 | One ironic or dramatic line — what's at stake for the user |
| `difficulty` | S2, S3 | AI judges: Easy = surface social, Medium = class/relationship tension, Hard = deep emotional stakes |
| `situationSetupForUser` | S1, S3, S4 | AI writes immersive 2nd-person Hinglish paragraph ending on the user's decision moment |
| `primalHook` | S2 | Direct — compressed into one sharp sentence |
| `atmosphere` | S2, S3 | AI writes 2–3 lines on the emotional air in the scene |
| `settingDescription` | S1, S4 | AI writes where she is, what she's doing, sensory details |
| `imagePrompt` | S1, S4, C6 | AI builds scene generation prompt: location + time + lighting + character physical vibe |
| `learningObjective` | S2, S5 | AI extracts the one social skill this scenario teaches |
| `goodOutcome` | S5 | Direct translation of arc destination into a one-line reward |
| `badOutcome` | S2 | Inversion of primal fear into a one-line consequence |
| `overallArc` | S5 | Direct — one sentence narrative journey start → end |
| `tone` | S3, S4 | AI picks a compound descriptor: playful-guarded / dry-curious / warm-charged / tense-nostalgic |
| `timeOfDay` | S4 | Direct extraction + time range formatting |
| `initialMessages` | S1, S3, C4 | AI writes 3 opening lines in the character's voice — varied energy, no two the same |
| `initialChips` | S2, S5 | AI writes 4 reply suggestions: one funny / one direct / one curious / one bold |
| `imageUrl` | imagePrompt | Image generation pipeline, stored URL |
| `chapterId` | — | Assigned by creator / content structure separately |
| `characterId` | — | Selected by creator — which character does this scenario belong to |

---

## PART 3 — SCENARIO BEAT CREATION

Beats are generated **after** the Scenario is created.
The LLM receives the full scenario context and generates all beats in sequence.

### User Inputs (3 fields for the full beat arc)

---

#### INPUT B1 — Number of Beats
```
Label:   "How many beats does this scenario have?"
Type:    Number (3–7 recommended)
Guidance:
  3 beats = short, punchy scenario (Easy difficulty)
  5 beats = standard arc with a real test moment (Medium)
  7 beats = full emotional journey with resist + break (Hard)
```

---

#### INPUT B2 — Beat Type Sequence
```
Label:   "Select the beat types in order"
Type:    Ordered multi-select (drag to reorder)

Beat Types:
  HOOK    → First impression. She's filtering. Skeptical by default.
  BUILD   → Warming up. Something small is working. Still testing.
  DEEPEN  → Real conversation starts. She's actually interested now.
  TEST    → She throws a challenge, goes quiet, or checks his reaction.
  PEAK    → Emotional high point. Something real has been said.
  RESIST  → She pulls back. Internal conflict. The situation's weight hits.
  BREAK   → Something cracks. An honest moment neither expected.

Example sequence (Medium, 5 beats):
  HOOK → BUILD → DEEPEN → TEST → PEAK

Example sequence (Hard, 7 beats):
  HOOK → BUILD → TEST → DEEPEN → RESIST → PEAK → BREAK
```

---

#### INPUT B3 — The Test Moment (for TEST beat only)
```
Label:   "What does she do to test him? What is she actually
          checking for?"
Type:    2 sentences, free text
Example: "She brings up Rohit casually mid-conversation.
          She's checking if he gets uncomfortable or honest."
Example: "She goes quiet for a beat after something real
          is said. She's checking if he fills the silence
          with something genuine or panics."
```

**Why it's required:**
Every other beat can be fully generated from scenario context.
The TEST beat needs a human creative decision because it is
the most character-specific, relationship-specific moment.
It cannot be safely inferred.

---

### DB Field Derivation Map — ScenarioBeat

| DB Field | Source Inputs | Derivation Logic |
|---|---|---|
| `beatNumber` | B1, B2 | Sequential from beat type order |
| `beatType` | B2 | Direct from sequence selection |
| `narrativeContext` | B2 + full scenario context | AI writes 2–3 lines: where is the scene emotionally at this beat |
| `characterEmotionalState` | S3 (Beat 1) + arc progression | AI tracks emotional journey from S3 toward S5 across beats |
| `flowDirective` | B2, S5, C5 | AI writes: what she does when user is doing well — in her voice, specific to her character |
| `hookDirective` | B2, S2, C5 | AI writes: how she redirects without cruelty when user is doing poorly |
| `minTurnsInBeat` | B1, difficulty | AI assigns: HOOK=2, BUILD=3, DEEPEN=3, TEST=2, PEAK=2, RESIST=2, BREAK=1 |
| `engagedAdvanceScore` | difficulty, beatType | AI assigns 1–5 threshold: harder beats require higher engagement to advance |
| `scenarioId` | — | Auto-linked to parent scenario |
| `characterId` | — | Auto-linked from scenario's characterId |

---

## Summary — Total Human Inputs Required

| Phase | Inputs | Fields Generated |
|---|---|---|
| Character | 6 inputs | 15 DB fields |
| Scenario | 5 inputs | 17 DB fields |
| Beats | 3 inputs | 9 DB fields × N beats |
| **Total** | **14 inputs** | **40+ DB fields** |

---

## The Prompt Engineering Principle Behind This

Every input above is chosen because it passes one test:

> **If two creators give different answers to this input,
> do they get dramatically different DB fields?**

- Different city → different dialect, references, class texture → different everything
- Different primal fear → different atmosphere, hook, outcomes, beat directives
- Different signature behavior → different voicePrompt, speakingStyle, textingSpeed

If an input doesn't pass this test — if swapping the answer doesn't change
the output meaningfully — it does not belong in the form.
That is why fields like `isActive`, `createdAt`, `chapterId` are not inputs.
And that is why `archetype phrase` and `specific trigger detail` are
the two most important inputs in the entire workflow.
