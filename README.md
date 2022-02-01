Social Backbones
==================

Accompanying code for Correia, Barrat & Rocha (2022).


Papers
------

- R.B. Correia, A. Barrat, L.M. Rocha [2022] "[The metric backbone preserves community structure and is a primary transmission subgraph in contact networks]()". Submitted.

- T. Simas, R.B. Correia, L.M. Rocha [2021]. "[The distance backbone of complex networks](https://academic.oup.com/comnet/article/9/6/cnab021/6403661)". *Journal of Complex Networks*, 9 (**6**):cnab021. doi: 10.1093/comnet/cnab021

- T. Simas and L.M. Rocha [2015]."[Distance Closures on Complex Networks](http://www.informatics.indiana.edu/rocha/publications/NWS14.php)". *Network Science*, 3(**2**):227-268. doi:10.1017/nws.2015.11

See the [`distanceclosure`](https://github.com/rionbr/distanceclosure) Python package for the computation of the metric backbone.


Color coding
==============

SocioPatterns
--------------


### Primary School

Border width: 20
Node border color: `#999999`
Node fil color
```
	Teachers:'#4c4c4c' #gray
	'1A':'#009999' # Cyan
	'1B':'#66ffff'
	'2A':'#004c00' #Green
	'2B':'#66b266'
	'3B':'#996300' # Orange
	'3A':'#ffc966'
	'4A':'#000099' # Blue
	'4B':'#6666ff'
	'5A':'#990000' # Red
	'5B':'#ff6666'
```
Edge opacity: 40% 


### High School

Colors:
```
	'2BIO1':'#ff0000' #red
	'2BIO2':'#ff6666'
	'2BIO3':'#990000'
	'MP'   :'#0000ff' #Blue
	'MP*1' :'#00007f'
	'MP*2' :'#7f7fff'
	'PC*'  :'#004c00' #Green
	'PC'   :'#66b266'
	'PSI*' :'#ffa500' # Orange
```

### Hospital

Colors:
```
	PAT:'Crimson'
	NUR:'Orange'
	MED:'DarkBlue'
	ADM:'ForestGreen'
```

### Workplace (large network)

Colors:
```
	DMI : '#FFA500' #Orange
	DSE : '#228B22' #ForestGreen
	DMCT: '#4B0082' #Indigo
	DST : '#A0522D' #Siena -> Brown(ish)
	DISQ: '#6b8e23' #OliveDrab
	SFLE: '#DC143C' #Crimson
	DCAR: '#C71585' #MediumVioletRed
	SSI : '#32CD32' #LimeGreen
	SRH : '#00008B' #DarkBlue
	SCOM: '#7FFFD4' #AquaMarine
	SDOC: '#FFA07A' #LightSalmon
	DG  : '#FFFF00' #Yellow
```

Toth
------

### Middle School

Colors:
```
	'7':'DarkBlue'
	'8':'Crimson'
	'null':'Orange'
```

### Elementary School

Colors:
```
	'K-0': #72e5be
	'K-1': #7FFFD4
	'K-2': #8bffd8
	
	'1-3': #b29600
	'1-4': #FFD700
	'1-5': #ffe34c
	
	'2-6': #176117
	'2-7': #228B22
	'2-8': #64ad64
	
	'3-9': #b26200
	'3-10': #FF8C00
	'3-11': #ffae4c
	
	'3-12': FFB6C1
	'4-12': FFB6C1
	
	'4-13': #b200b2
	'4-14': #ff4cff
	
	'5-15': #000061
	'5-16': #4c4cad
	'5-17': #00008B
	
	'6-18': #9a0e2a
	'6-19': #DC143C
	'6-20': #e65a76


```
Salathé
-----------

Colors:
```
	students:'DarkBlue',
	teacher:'Crimson',
	staff:'DarkOrange',
	other:'DarkGreen',
```

Acknowledgments
------------------

We thank [Nathan Ratkiewicz](https://github.iu.edu/nratkiew) for commits in early versions of this code.
