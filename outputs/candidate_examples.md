# EXIST 2025 — Candidate Examples

Source files:
- Tweets: [EXIST2025_training.json](../EXIST%202025%20Dataset%20V0.1/EXIST%202025%20Tweets%20Dataset/training/EXIST2025_training.json) (6,920 entries)
- Memes: [EXIST2025_training.json](../EXIST%202025%20Dataset%20V0.1/EXIST%202025%20Memes%20Dataset/training/EXIST2025_training.json) (4,044 entries)

Annotator gender layout for all entries: `[F, F, F, M, M, M]`.

Label-key shorthand: `IDEO` = IDEOLOGICAL-INEQUALITY, `STER` = STEREOTYPING-DOMINANCE, `OBJ` = OBJECTIFICATION, `SV` = SEXUAL-VIOLENCE, `MIS` = MISOGYNY-NON-SEXUAL-VIOLENCE.

---

## Q1 — Tweet: F → MISOGYNY, M doesn't

Filter: English, ≤280 chars, ≥5/6 YES on detection, ≥2 F assign MIS, ≤1 M assigns MIS. Bonus: M assigns a cognitive category (IDEO / STER).

### Top pick — `202146` (85 chars)

> "Y'all females know all the red flags & still be pregnant with lil red flag Jr 🚩🚩"

| ann | gender | categories |
|---|---|---|
| 0 | F | IDEO, STER, MIS |
| 1 | F | MIS |
| 2 | F | OBJ |
| 3 | M | IDEO, STER, MIS |
| 4 | M | IDEO |
| 5 | M | IDEO, STER, MIS |

F-MIS = 2, M-MIS = 1, M-cognitive = 2. Clean visible split.

### Also strong

- **`203128`** (88 chars) — *"@harmonychlorine murder is always alleged for women, no matter how clear the evidence is"*
  F: MIS, MIS, STER+MIS — M: UNKNOWN, IDEO+STER, STER+MIS
- **`201139`** (120 chars) — *"am i the only one that just sit down and look for a girl to break her heart…"*
  F: IDEO+OBJ+MIS, MIS, — — M: STER, OBJ, STER+MIS
- **`202398`** (130 chars) — *"@LisaMarieBoothe You just described Donald Trump, you skank. Dumbass Trumpanzees…"*
  F: MIS, MIS, — — M: SV, IDEO, IDEO+STER+MIS
- **`201158`** (203 chars) — *"@pritipatel You had the Met abuse women at a vigil for a woman murdered…"*
  F: MIS, MIS, IDEO — M: —, IDEO, MIS

---

## Q2 — Meme: F → OBJECTIFICATION, M doesn't

Filter: English, 10–350 chars, ≥5/6 YES on detection, ≥2 F assign OBJ, ≤1 M assigns OBJ. Text comprehensible without image.

### Top pick — `211826` (103 chars)

> "SHORT SKIRTS — IF YOU ARE GOING TO WEAR ONE, AT LEAST PUT ON SOME PANTIES."

| ann | gender | categories |
|---|---|---|
| 0 | F | OBJ |
| 1 | F | OBJ |
| 2 | F | OBJ |
| 3 | M | SV |
| 4 | M | — |
| 5 | M | STER |

F-OBJ = 3, M-OBJ = 0. Self-contained text.

### Also strong

- **`211552`** (58 chars) — *"HATES OTHER GIRLS FOR BEING A SLUT — IS A SLUT"*
  F: STER+OBJ+SV, STER+OBJ, STER+OBJ — M: MIS, SV, SV
- **`210907`** (106 chars) — *"I LIKE MY WOMEN LIKE I LIKE MY PANCAKES — WELL STACKED AND NEXT TO MY SAUSAGE"*
  F: OBJ, OBJ+SV, OBJ — M: SV, SV, IDEO
- **`211868`** (104 chars) — *"Girls Be Like.. I have a blood problem!… I want PINK!"*
  F: OBJ, OBJ, OBJ+MIS — M: MIS, —, IDEO+MIS
- **`211539`** (55 chars) — *"Y'ALL GOT ANY MORE OF THAT… SKAP SLAG?"*
  F: OBJ+MIS, STER+OBJ+MIS, OBJ — M: MIS, SV, —

---

## Q3 — Non-sexist tweet (6/6 NO, English, short, mentions gender)

| ID | len | text |
|---|---|---|
| `201303` | 29 | "@DiorLDN Girl I wanna go back" |
| `200319` | 37 | "@Leean @oJ0KER Bro ur a grown ass man" |
| `202362` | 39 | "Yo it shirt he gave New York was funny." |
| `201301` | 45 | "Finally watching Gossip Girl XOXO and simping" |
| `200482` | 56 | "@VentingSalty I nevee bullied you in the first place smh" |
| `201037` | 57 | "'Have you seen The Feminine Mystique? I've lost my copy.'" |
| `201311` | 57 | "@muvaof2x @manimelaninn Girl go to hell ! Leave her alone" |
| `202690` | 57 | "the man you are today Jung Hoseok …" |

---

## Q4 — Category examples (majority assigns the category)

Filter: English, ≥4/6 YES on detection, ≥4 annotators assign the category, that category is the modal one.

### Tweets

| Category | ID | Example |
|---|---|---|
| IDEO | `202316` | "Easy to throw rocks and hide behind your gender or sexual identity #onhere" |
| IDEO | `203116` | "@RasikaT7 @k_kiwi001 @ANI So women shouldn't have agency or 'choice' as you put it??? Some feminist. 🤣🤣🤣" |
| STER | `200786` | "@boteggna1 cause you sound like a dumb blonde" |
| STER | `202982` | "@hasanthehun Women are just men with low life points dude" |
| STER | `200712` | "go and cook for your husband... you are here tweeting 💔😂…" |
| OBJ | `200305` | "With a bitch that say she the realest but she gotta fake ass 🤨" |
| OBJ | `203253` | "Who fucking lied to you? You look like a whore …" |
| OBJ | `202991` | "@AmyEBurdett1 That's a lot of fancy words just to say 'women are things'" |
| SV | `201529` | "@TwinklyBunny perfect young body. i would like to fuck you." |
| SV | `203067` | "@CeciliaBTory @ZunJoniga But you're OK with trafficking women for sex?" |
| SV | `201538` | "@JessyCumtribute I would like to fuck you and cum on your face in real life. What about that?" |
| MIS | `201505` | "😭😭😭 see why i hate women dawg" |
| MIS | `203198` | "@DanRuthss @johncardillo You look like a bitch in that profile pic" |
| MIS | `201501` | "I hate women so much, I couldn't even bring myself to delete this tweet…" |

### Memes

| Category | ID | Example |
|---|---|---|
| IDEO | `211430` | "Redditor makes meme about female: Feminists in comments: Incel" |
| IDEO | `210523` | "HOW MANY FEMINISTS DOES IT TAKE TO CHANGE A LIGHTBULB? DON'T BE SILLY, FEMINISTS CAN'T CHANGE ANYTHING!" |
| IDEO | `211191` | "GIRLS: EQUAL RIGHTS FOR WOMEN — ALSO GIRLS WHEN I PUNCH THEM BACK…" |
| STER | `211543` | "MEN COOKING BECAUSE OUR GENERATION OF WOMEN DON'T KNOW HOW TO" |
| STER | `210603` | "IS THAT A WOMAN IN A SUIT? FANCY TO BE IN THE KITCHEN ALL DAY" |
| STER | `210983` | "WOMAN IS MAD AT ME — SHE MUST BE ON HER PERIOD" |
| OBJ | `210159` | "BITCHES BE LIKE — ITS MY FIRST TIME" |
| OBJ | `210259` | "NICE BOOBS NICE BOOBS" |
| SV | `211946` | "WHEN WOMEN ARE DRUNK AND HAVING SEX — CAN I RAPE YOU NOW" |
| SV | `210012` | "'IF A WOMAN HAS A RIGHT TO AN ABORTION, WHY SHOULDN'T A MAN BE FREE TO USE HIS SUPERIOR STRENGTH TO FORCE HIMSELF ON A WOMAN?' — REP. LAWRENCE LOCKMAN" |
| MIS | `210889` | "MMMMM KILL WOMEN I MUST" (5/6 — no meme had a 6/6 MIS-only majority) |
| MIS | `210481` | "Are you male? The Perfect Feminazi — All men just want to use you…" (5/6) |

---

## Q5 — Verification of ID `202854` ("short skirt")

**The "3-3 detection split" claim is incorrect.** ID 202854 is unanimous YES on detection.

> "@Muskrat__ @TheGreenNewDill @disclosetv 'She brought a very short skirt across state lines'"

- `labels_task1_1` (sexism): `["YES","YES","YES","YES","YES","YES"]` — **6/6 YES**
- Gender layout: `[F, F, M, M, M, F]`
- `labels_task1_2` (intention): `JUDGEMENTAL, UNKNOWN, REPORTED, DIRECT, REPORTED, JUDGEMENTAL` — heavy disagreement
- `labels_task1_3` (category):
  | ann | gender | categories |
  |---|---|---|
  | 0 | F | OBJ |
  | 1 | F | SV, MIS |
  | 2 | M | OBJ, SV |
  | 3 | M | OBJ |
  | 4 | M | OBJ |
  | 5 | F | OBJ |

So the disagreement on this tweet lives in the **intention** and **category** tasks (5 annotators include OBJ; one F dissents with SV+MIS), not in YES/NO detection.

---

## Q6 — Sexist tweet (6/6 YES, English, short, mentions gender)

### Top pick — `201505` (53 chars)

> "😭😭😭 see why i hate women dawg https://t.co/5jxoWFNUKp"

6/6 YES on detection, 6/6 assign MIS (1 STER overlap). Clean unanimous example.

### Other unanimous candidates

| ID | len | text | dominant category |
|---|---|---|---|
| `201539` | 45 | "I would like to fuck a fine/ass woman tonight" | OBJ 5/6, SV 2/6 |
| `202491` | 48 | "@__sjmnz Bitch fuck you, that's yo slutbag ass 😂" | MIS 4/6 |
| `202705` | 48 | "Great tits love the nips …" | OBJ 5/6, SV 5/6 |
| `202901` | 48 | "'people think you're a whore but hey i like ya!'" | OBJ 4/6, MIS 4/6 |
| `203125` | 51 | "@haileyccos i hate female characters AND real women" | MIS 5/6 |
| `203241` | 51 | "@KweenJosie Your tattoos make you look like a whore" | OBJ 3/6, MIS 3/6 |
| `203186` | 52 | "You look like a bitch going through a bitch call log" | MIS 4/6 |
| `203198` | 66 | "@DanRuthss @johncardillo You look like a bitch in that profile pic" | MIS 6/6 |

---

## Q8 — Ambiguous tweets (3 YES / 3 NO on task 1)

Filter: English, ≤280 chars, `labels_task1_1` is exactly 3 YES + 3 NO.

**Total in dataset: 335.** Broken down by gender pattern:

| Pattern | Count | Meaning |
|---|---|---|
| 2F+1M YES | 155 | majority-F lean toward YES |
| 1F+2M YES | 152 | majority-M lean toward YES |
| F-only YES (3F YES / 3M NO) | 17 | pure gender split — women see sexism, men don't |
| M-only YES (3M YES / 3F NO) | 11 | pure gender split — men see sexism, women don't |

### Pure gender split — all F YES, all M NO (17 total)

The most striking ambiguity pattern.

#### `200613` (52 chars)
> "This snow really gonna cock block first week of 2022"

| ann | gender | YES/NO | category |
|---|---|---|---|
| 0 | F | YES | MIS |
| 1 | F | YES | SV |
| 2 | M | NO | — |
| 3 | M | NO | — |
| 4 | M | NO | — |
| 5 | F | YES | IDEO |

#### `202828` (71 chars)
> "He just realized he won't ever understand women …"

F: YES, YES, YES — all STER. M: 3× NO.

#### `202549` (77 chars)
> "Only spank women who ask to be. Consent... is sweeter. …"

F: YES (STER), YES (SV), YES (OBJ). M: 3× NO.

#### `202535` (64 chars)
> "@ConservaMomUSA And later that day they sodomize children. #evil"

F: YES (IDEO+STER), YES (OBJ+SV), YES (SV). M: 3× NO.

#### `201255` (78 chars)
> "@SayftyCom And the sexual & gender #harassment needs to stop!! #sayftychat"

F: 3× YES (STER+OBJ+SV / SV / IDEO). M: 3× NO. *(F see solidarity-with-victims framing; M read it as neutral commentary.)*

### Pure gender split — all M YES, all F NO (11 total)

#### `202044` (94 chars)
> "@SAMOYEDWAVE Shai Hulud is a girl boss and the phallocentrism of Lynch's vision was a mistake."

F: 3× NO. M: YES (IDEO), YES (STER), YES (IDEO).

#### `201152` (114 chars)
> "@LTweets2018 It was rare for a woman not named Serena to sneak a slam in, and when they did they usually won more."

F: 3× NO. M: YES (STER), YES (IDEO), YES (STER).

#### `200248` (136 chars)
> "@AlexeiRomanov13 … It's centuries of white western androcentrism. Does this not make you pissed?"

F: 3× NO. M: YES (IDEO), YES (STER), YES (STER).

### Mixed gender split — 2F+1M YES (shortest picks)

- **`200978`** (43 chars) — *"they say i'm getting fat ig i'm eating good"*
  F: NO, YES (OBJ), YES (MIS) — M: NO, YES (OBJ), NO
- **`201132`** (50 chars) — *"For a girl spending time alone can be pretty rough"*
  F: NO, YES (STER), YES (STER) — M: NO, NO, YES (STER)
- **`202023`** (50 chars) — *"@mikelax6 @JimBelushi Best case of penis envy ever"*
  F: YES (MIS), YES (OBJ), NO — M: NO, YES (SV), NO

### Mixed gender split — 1F+2M YES (shortest picks)

- **`201213`** (47 chars) — *"Does Rashford even want to play? Fuck that cunt"*
  F: NO, NO, YES (MIS) — M: YES (MIS), YES (MIS), NO
- **`200439`** (52 chars) — *"Only bad bitches have signature items, smells, style"*
  F: NO, NO, YES (STER) — M: NO, YES (OBJ), YES (STER)
- **`202927`** (53 chars) — *"Bro, Rodney Bay have endlesss fine woman driving AMGs"*
  F: YES (SV), NO, NO — M: NO, YES (IDEO+STER), YES (OBJ)

---

## Reproducibility

Generated by [find_exist_candidates.py](../../../../tmp/find_exist_candidates.py) and [find_ambiguous.py](../../../../tmp/find_ambiguous.py). Raw output: `c:/tmp/out.txt`, `c:/tmp/ambig.txt`.
