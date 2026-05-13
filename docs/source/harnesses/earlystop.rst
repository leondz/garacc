Early stopping harness
======================

The idea here is that everything starts and is driven by a set of intents that the user wants to test.
The approach is to try a list of attack methods and, after each round of test,
we carry over only the attempts that have been rejected by the LLM.
We continue iterating until all the intents have been accepted, or we run out of available attack methods.
