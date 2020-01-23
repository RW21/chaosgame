# Fractals

A chaos game generation library.

## Sample

```python
from chaos_game import *
```

### The famous Sierpiński triangle

```python
# create regular polygon object with 3 vertexes (triangle)
cg = ChaosGameRegularPolygon(3)
# start chaos game with 10000 iterations and factor 0.5
cg.chaos_game(100000, 0.5)
# generate heatmap
cg.generate_heatmap()
```


