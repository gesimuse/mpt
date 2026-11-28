// POV reaction skits (2026-11-28: the user on the first three, "not bad" / "yeah nice"): she reads a
// message on her phone, reacts, then one short line to camera. Hand-written so every one is a real
// joke; nothing about money (2026-11-24). `react` is what her face does before she talks.
// Then "add jargon and maybe curse words in the hook" / "not written": slang in the on-screen tag, swears
// only SAID in her line (the captions mask them, persona/talk.py _mask).
const READ = "she glances down at her phone screen reading a message, ";
export const POV_SKITS = [
  ["POV: bro texts \"wyd\" at 2am like it's normal", "raises one eyebrow and gives a small smirk", "Wyd? At two a.m.? Bro, go the hell to sleep."],
  ["POV: he says he's \"not like other guys\"", "then slowly rolls her eyes with a little smile", "Babe, that's literally what every other guy said. Be so for real."],
  ["POV: he actually made a real plan for the date", "her eyebrows lift and a surprised smile spreads", "Wait, you planned something? Okay, lowkey that's hot."],
  ["POV: you sent a paragraph and he replied \"k\"", "her smile drops into a flat, unimpressed stare", "I wrote you a whole ass essay and you gave me one letter. The audacity."],
  ["POV: he remembered the tiny thing you said last week", "her face softens into a slow, warm smile", "You remembered that? Okay, no cap, you have my attention."],
  ["POV: he sends a meme instead of saying sorry", "presses her lips together, trying not to laugh", "Funny. Still pissed. But funny."],
  ["POV: \"I'll call you later\" was three days ago", "tilts her head with a fake sweet smile", "Later is a time, babe, not a whole ass lifestyle."],
  ["POV: he complimented her laugh, not her body", "blushes and looks away for a second, smiling", "Okay, that one actually got me. Damn."],
  ["POV: he watched her story and left her on read", "narrows her eyes slowly", "So you saw it and chose silence. Bold. Red flag, but bold."],
  ["POV: \"you up?\" at 3am. again.", "laughs once and shakes her head", "I'm up. Just not for your bullshit."],
  ["POV: he says she's \"not like other girls\"", "squints, half amused", "Not like other girls how? Choose your next words carefully, bro."],
  ["POV: he texted good morning without being asked", "breaks into a sleepy, happy smile", "A good morning text? The bar was on the floor, but damn, you cleared it."],
  ["POV: you ask about plans and he says \"we'll see\"", "slowly raises both eyebrows", "We'll see means hell no, and we both know it."],
  ["POV: he double texted and owned it", "smiles with mock surprise, hand on her chest", "A double text? Main character energy. I respect it."],
  ["POV: \"hey stranger\" after ghosting for two months", "stares blankly, then smirks", "Stranger is right. Stay that way, babe."],
  ["POV: he actually listened to her whole rant", "looks up, genuinely touched, then smiles", "You listened to all of it? Okay, I'm lowkey obsessed."],
  ["POV: he said \"calm down\" mid-argument", "freezes, then gives a slow dangerous smile", "Oh, I'm calm now. Scary calm. Run."],
  ["POV: he picked her up for the date like a grown man", "gasps softly with a big smile", "You picked me up? Who raised you? Tell them thank you."],
  ["POV: one-word answers all night", "yawns dramatically", "Talking to you is like texting a wall. A cute wall, but still."],
  ["POV: he asked about her day and actually meant it", "pauses, then smiles slowly", "No one asks that and waits for the answer. Okay, green flag."],
  ["POV: he said \"you look tired\"", "blinks slowly, unimpressed", "I did my makeup for forty damn minutes. Try again."],
  ["POV: he showed up on time. like, actually on time", "checks the time, then looks up impressed", "On time? Okay, rizz unlocked."],
  ["POV: he says \"I'm just bad at texting\"", "gives a tight polite smile", "You're not bad at texting. You're bad at texting me. Big difference."],
  ["POV: he noticed she changed her hair", "touches her hair and lights up", "You noticed? Okay, you can stay. For now."],
  ["POV: \"let's just see where it goes\"", "nods slowly with a fake smile", "Where it goes is a situationship. I know this road, bro."],
  ["POV: he put his phone away on the date", "looks surprised, then smiles warmly", "Phone away? That's the hottest shit you did tonight."],
  ["POV: he sent a voice note singing to her", "covers her mouth, laughing", "That was terrible. Send another one."],
  ["POV: he says \"you're overthinking it\"", "stares, then takes a breath", "I'm thinking the right amount about the wrong damn guy."],
  ["POV: he made her laugh on the first date", "smiles and bites her lip", "Funny and cute? That's dangerous as hell. Second date."],
  ["POV: he asked for her number in person, no DMs", "raises her eyebrows, impressed", "No DM, no games, just asked? Okay, here. Don't fuck it up."],
].map(([label, react, line]) => ({ label, react: READ + react, line }));
