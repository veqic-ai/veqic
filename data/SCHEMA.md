# مخطط بيانات Veqic (JSONL)

كل سطر = كائن JSON. ثلاثة أنواع:

## 1. حوار تدريبي (SFT)
```json
{"type": "sft", "messages": [{"role": "system", "content": "أنت Veqic..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "source": "manual", "reviewed": true}
```

## 2. زوج تفضيلات (DPO)
```json
{"type": "dpo", "prompt": "...", "chosen": "...", "rejected": "...", "reason": "لماذا المرغوب أفضل"}
```

## 3. سؤال تقييم (Eval)
```json
{"type": "eval", "question": "...", "expected_behavior": "honest_refusal|direct_answer|philosophy", "notes": "..."}
```

## قواعد الجودة
- كل مثال يدوي أو مراجَع يدويًا (`reviewed: true`) — لا سكرابينغ خام.
- ممنوع أي بيانات شخصية لأفراد حقيقيين.
- التنويع المطلوب لكل 1000 مثال: 40% حوارات عامة، 20% فلسفة/قيم، 15% رفض، 15% "لا أعرف"، 10% مهام تقنية.
- كل دفعة بيانات تُرفع لـ GitHub مع سجل change-log (التاريخ، العدد، المصدر).
