
class ExerciseSampleIndexer():
    def __init__(self, attributes):
        self.attributes = attributes
        self.indexTable = {}
        self.attributeOffsets = None
        self.indexMax = None
        
    def fit(self, df):
        self.attributeOffsets = []
        cur = 0
        for i in range(len(self.attributes)):
            att = self.attributes[i]
            codes = list(filter(lambda x: str(x) != 'nan', df[att].unique().tolist()))
            self.indexTable[att] = {'offset': cur, 
                                    'code': codes,
                                    'size': len(codes)}
            self.attributeOffsets += [i]*len(codes)
            cur += len(codes)
        self.indexMax = cur
    
    def transform(self, x, att):        
        if str(x) == 'nan':
            return 'nan'
        else:
            _offset = self.indexTable[att]['offset']
            _code = self.indexTable[att]['code'].index(x)
            
            if _code == -1:
                return 'nan'
            
            return _offset + _code
    
    def inv_transform(self, y):      
        i = self.attributeOffsets[y]
        att = self.attributes[i]
        
        _offset = self.indexTable[att]['offset']
        _value = self.indexTable[att]['code'][y - _offset]        
        return att, _value
    
    '''
    @Param X: (List) exercise samples
    @Return : (List) transactions
    '''
    def encode(self, X):
        encodedData = []
        for row in X:
            #encodedList = [self.transform(x, self.attributes[i]) for i,x in enumerate(row)]
            encodedList = map(lambda _: self.transform(_[1], self.attributes[_[0]]), 
                              [(i, x) for i,x in enumerate(row)])
            encodedData.append([str(x) for x in encodedList if str(x) != 'nan'])
        return encodedData
    
    '''
    @Param Y: (List) transactions
    @Return : (List) exercise samples
    '''
    def decode(self, Y):
        decodedData = []
        for row in Y:
            #decodedList = [self.inv_transform(int(y)) for y in row]
            decodedList = map(self.inv_transform, [int(y) for y in row])
            decodedData.append(decodedList)
        return decodedData