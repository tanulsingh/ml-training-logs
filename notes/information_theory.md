The thing Olah is leaning on is the prefix code rule: no codeword can be a prefix of another codeword. Once you understand that, the "sacrifice" idea snaps into place.

Why prefix codes matter

Suppose you used 0 for "dog" and 01 for "cat". Now the decoder sees bits coming in: 0...

When it reads the first 0, does that mean "dog" already? Or should it wait for the next bit in case it's 01 = "cat"? Ambiguous. The decoder can't tell.

So the rule is: no codeword can be the start of another codeword. This is what makes a code uniquely decodable bit-by-bit, no commas needed.

Visualizing it as a binary tree

Picture the space of all possible bit strings as a tree:

                    root
                  /      \
                  0        1
                /   \    /   \
              00   01  10   11
              /\   /\   /\   /\
            000 001 010 011 100 101 110 111
            ...

Going left = 0, going right = 1. Every node corresponds to a bit-string (its path from the root).

A codeword is a node. And the prefix rule says: once you pick a node as a codeword, you can't use any node beneath it — because every node beneath has your codeword as a
prefix.

The sacrifice

Now look at how much you lose by claiming nodes at different depths:

- Claim node 0 (depth 1, 1-bit codeword): you kill the entire left subtree. That's 00, 01, 000, 001, 010, 011, ... → half of all possible codewords gone.
- Claim node 01 (depth 2, 2-bit codeword): you kill everything starting with 01 → 010, 011, 0100, ... → a quarter of all possible codewords gone.
- Claim node 0101 (depth 4, 4-bit codeword): you kill only 1/16 of the tree.

Pattern: a codeword of length $L$ sacrifices a fraction $1/2^L$ of the codeword space.

So shorter codeword → bigger fraction sacrificed.

Why this forces a trade-off

You have a fixed budget of "codeword space" — the whole tree, which sums to 1. Every codeword you pick takes a slice of it ($1/2^L$). The total has to fit:

$$\sum_x \frac{1}{2^{L(x)}} \leq 1$$

This is Kraft's inequality — the formal statement of the trade-off Olah is describing.

If you spend a lot of your budget making one codeword short (claiming a high node in the tree, killing lots of descendants), there's less budget left for the others, so they
  have to be longer (deeper in the tree, where each slice is small).

Concrete example

Try giving 4 words all 1-bit codes:
- Claim 0 → sacrifices 1/2
- Claim 1 → sacrifices the other 1/2
- Budget used: 1/2 + 1/2 = 1. Full. Can't add a third codeword without ambiguity.

So 4 words can't all be 1-bit. You're forced into longer codes for some of them.

Now Olah's setup — dog (1 bit), cat (2 bit), fish (3 bit), bird (3 bit):
- 0 costs 1/2
- 10 costs 1/4
- 110 costs 1/8
- 111 costs 1/8
- Total: 1/2 + 1/4 + 1/8 + 1/8 = 1.0. Exactly fills the tree.

That code is optimal in the sense that there's no room left — every dropped slice has been claimed.

The connection to Shannon

The optimal trade-off — what length to give each codeword — turns out to be: length proportional to −log₂ p(x). Common words ($p$ large) → small −log₂ p → short codeword
(small sacrifice from the budget is fine because you'll save on every transmission). Rare words → long codeword (big sacrifice would be wasteful for something you rarely
send).

This is exactly the formula the LLM loss uses, and it's why "shorter codewords cost more" matters: the optimization isn't "make everything short," it's "spend your budget on
  the words you'll send most."