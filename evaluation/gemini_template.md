# Gemini Flash — Manual Completion Template

**No Gemini API key is used anywhere in this repo.** This file is a manual
paste-in template: `comparison_table.py` reads it to build the 3-model
comparison table, and will clearly mark the Gemini column `PENDING` for any
prompt whose completion field below is still empty.

## How to fill this in

1. Open the Gemini web UI ([gemini.google.com](https://gemini.google.com))
   or Google AI Studio, and select the **Gemini Flash** model.
2. For each prompt below, paste the *exact* prompt text (nothing added, no
   extra instructions) into a fresh/empty chat, using default settings
   (default temperature, no system prompt).
3. Copy Gemini's full response and paste it into the fenced code block
   under that prompt, replacing the `(PASTE COMPLETION HERE)` placeholder.
   Do not truncate or edit the response — paste it as returned, even if
   it's much longer or more coherent than 150 tokens (that contrast is
   part of the analysis).
4. Save this file, then re-run `python comparison_table.py`.

---

## Prompt 1: "To be, or not to "

```
be, that is the question: Whether 'tis nobler in the mind to suffer The slings
and arrows of outrageous fortune, Or to take arms against a sea of troubles, And
by opposing end them?
```

---

## Prompt 2: "O Romeo, Romeo, wherefore art thou "

```
This is a well-known line spoken by Juliet in William Shakespeare's play, Romeo
and Juliet (Act II, Scene 2).

A common misconception is that Juliet is asking where Romeo is physically
located. However, in Elizabethan English, "wherefore" actually means "why."

Juliet is asking why Romeo must be who he is—specifically, a Montague, the
family that is the sworn enemy of her own family, the Capulets. She is lamenting
the family feud that stands in the way of their love.

The full passage continues:

"O Romeo, Romeo! wherefore art thou Romeo? Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love, And I'll no longer be a Capulet."
```

---

## Prompt 3: "Friends, Romans, countrymen, lend me your "

```
ears! I come to bury Caesar, not to praise him.
```

---

## Prompt 4: "Now is the winter of our discontent"

```
Now is the winter of our discontent" is the famous opening line of William Shakespeare's historical play "Richard III."
Spoken by the protagonist, Richard, Duke of Gloucester (who later becomes King Richard III), these lines reflect on the end of the Wars of the Roses and the beginning of a period of relative peace under his brother, King Edward IV.
Here is the context and the continuation of the famous soliloquy:
"Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried."
Meaning
"Winter of our discontent": Refers to the long, bitter period of civil war (the Wars of the Roses) that caused suffering and unhappiness.
"Made glorious summer": The war has ended, and the time of peace and prosperity has arrived.
"This sun of York": A pun on "son of York" (referring to King Edward IV) and the sun symbol on the House of York's coat of arms.
While the play begins with a celebration of peace, the speech quickly turns dark as Richard—who is physically deformed and bitter—reveals that because he cannot be a lover or a man of peace, he is "determined to prove a villain" and manipulate his way to the throne.
```
