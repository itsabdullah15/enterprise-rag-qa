def verify_faithfulness(verifier_llm, answer: str, context: str) -> bool:
    verification_prompt = f"""
        You are a strict factual verifier.

        Rules:
        - Break the answer into individual factual claims.
        - For EACH claim, check whether it is directly supported
        by the context (verbatim OR clear paraphrase).
        - If ANY claim is not supported by the context, reply NO.
        - Do NOT assume external knowledge.

        Context:
        {context}

        Answer:
        {answer}

        Respond ONLY with:
        YES or NO
        """
    
    print(context)

    print(answer)




    verdict = verifier_llm.generate(
            verification_prompt,
            temperature=0.0
        ).strip().upper()
    
    print(verdict)


    return verdict == "YES"
