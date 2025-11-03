*RSAA Project*
Residency Safe Answering Agent- a region-aware RAG concept where data stays local and only minimal signals move, so answers get composed with citations in the requester’s region. (No raw docs crossing borders.) 

Built it on Flower framework- US/EU/APAC, each ran as a Flower client wrapping its local retriever/index. A lightweight Flower server fanned out the question and collected only top-k snippet metadata (IDs + scores) from each client. We wrote a custom Flower Strategy to fuse rankings with RRF and return a single list of snippet refs—still no documents in transit. 
The response was then generated entirely in the requester’s region using those refs. Flower’s simulation flow lets us iterate locally on “three regions” and then point the same code to live nodes, making region add/remove nearly plug-and-play.
