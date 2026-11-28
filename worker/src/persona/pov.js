// POV reaction skits (2026-11-28: the user on the first three, "not bad" / "yeah nice"): she reads a
// message on her phone, reacts, then one short line to camera. Hand-written so every one is a real
// joke; nothing about money (2026-11-24). `react` is what her face does before she talks.
// Then "add jargon and maybe curse words in the hook" / "not written": slang in the on-screen tag, swears
// only SAID in her line (the captions mask them, persona/talk.py _mask).
const READ = "she glances down at her phone screen reading a message, ";
export const POV_SKITS = [
  ["POV: bro texts \"wyd\" at 2am like it's normal", "raises one eyebrow slightly", "Wyd? At two a.m.? Bro, go the hell to sleep."],
  ["POV: he says he's \"not like other guys\"", "then gives a tiny eye roll", "Babe, that's literally what every other guy said. Be so for real."],
  ["POV: he actually made a real plan for the date", "her eyebrows lift a little and a small smile starts", "Wait, you planned something? Okay, lowkey that's hot."],
  ["POV: you sent a paragraph and he replied \"k\"", "her face goes flat and unimpressed", "I wrote you a whole ass essay and you gave me one letter. The audacity."],
  ["POV: he remembered the tiny thing you said last week", "her face softens into a small smile", "You remembered that? Okay, no cap, you have my attention."],
  ["POV: he sends a meme instead of saying sorry", "presses her lips together, holding back a small laugh", "Funny. Still pissed. But funny."],
  ["POV: \"I'll call you later\" was three days ago", "tilts her head slightly with a tight polite smile", "Later is a time, babe, not a whole ass lifestyle."],
  ["POV: he complimented her laugh, not her body", "looks away for a second with a shy little smile", "Okay, that one actually got me. Damn."],
  ["POV: he watched her story and left her on read", "narrows her eyes slightly", "So you saw it and chose silence. Bold. Red flag, but bold."],
  ["POV: \"you up?\" at 3am. again.", "gives a short quiet laugh through her nose", "I'm up. Just not for your bullshit."],
  ["POV: he says she's \"not like other girls\"", "squints a little, half amused", "Not like other girls how? Choose your next words carefully, bro."],
  ["POV: he texted good morning without being asked", "gives a small sleepy smile", "A good morning text? The bar was on the floor, but damn, you cleared it."],
  ["POV: you ask about plans and he says \"we'll see\"", "raises both eyebrows slightly", "We'll see means hell no, and we both know it."],
  ["POV: he double texted and owned it", "looks a little surprised and amused", "A double text? Main character energy. I respect it."],
  ["POV: \"hey stranger\" after ghosting for two months", "stares blankly for a second", "Stranger is right. Stay that way, babe."],
  ["POV: he actually listened to her whole rant", "looks up, quietly touched, with a small smile", "You listened to all of it? Okay, I'm lowkey obsessed."],
  ["POV: he said \"calm down\" mid-argument", "goes still, then gives a small tight smile", "Oh, I'm calm now. Scary calm. Run."],
  ["POV: he picked her up for the date like a grown man", "looks pleasantly surprised with a small smile", "You picked me up? Who raised you? Tell them thank you."],
  ["POV: one-word answers all night", "blinks slowly, bored", "Talking to you is like texting a wall. A cute wall, but still."],
  ["POV: he asked about her day and actually meant it", "pauses, then a small smile", "No one asks that and waits for the answer. Okay, green flag."],
  ["POV: he said \"you look tired\"", "blinks once, unimpressed", "I did my makeup for forty damn minutes. Try again."],
  ["POV: he showed up on time. like, actually on time", "looks up, mildly impressed", "On time? Okay, rizz unlocked."],
  ["POV: he says \"I'm just bad at texting\"", "gives a tight polite smile", "You're not bad at texting. You're bad at texting me. Big difference."],
  ["POV: he noticed she changed her hair", "touches her hair with a small pleased smile", "You noticed? Okay, you can stay. For now."],
  ["POV: \"let's just see where it goes\"", "nods slightly with a tight smile", "Where it goes is a situationship. I know this road, bro."],
  ["POV: he put his phone away on the date", "looks mildly surprised, then a small warm smile", "Phone away? That's the hottest shit you did tonight."],
  ["POV: he sent a voice note singing to her", "holds back a laugh, smiling a little", "That was terrible. Send another one."],
  ["POV: he says \"you're overthinking it\"", "stares for a second, then a small breath", "I'm thinking the right amount about the wrong damn guy."],
  ["POV: he made her laugh on the first date", "gives a small smile", "Funny and cute? That's dangerous as hell. Second date."],
  ["POV: he asked for her number in person, no DMs", "raises her eyebrows slightly, impressed", "No DM, no games, just asked? Okay, here. Don't fuck it up."],
].map(([label, react, line]) => ({ label, react: READ + react, line }));
