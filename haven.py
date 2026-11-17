"""Haven: small persistent recurrent adaptive graph. Standard library only."""
import json, math, os, random, threading, time
from pathlib import Path

class Haven:
    def __init__(self, path="haven_state.json", seed=7):
        self.path=Path(path)
        self.lock=threading.RLock()
        self.rng=random.Random(seed)
        self.nodes={}
        self.edges={}
        self.history=[]
        self.steps=0
        self.learning_rate=0.12
        self.last_input=None
        if self.path.exists():
            self.load()
        else:
            for name in ("A","B","C","D"):
                self.add_node(name)
            self.save()

    def add_node(self, name):
        with self.lock:
            if name not in self.nodes:
                self.nodes[name]={"activation":0.0,"vector":[self.rng.uniform(-1,1) for _ in range(8)],"visits":0}
            return self.nodes[name]

    def key(self,a,b):
        return a+"\u2192"+b

    def weights_from(self,source):
        return {target:self.edges.get(self.key(source,target),0.0) for target in self.nodes if target!=source}

    def predict(self,source):
        with self.lock:
            self.add_node(source)
            scores=self.weights_from(source)
            if not scores: return None,{}
            # Softmax yields a normalized prediction distribution.
            peak=max(scores.values())
            exps={k:math.exp((v-peak)*3) for k,v in scores.items()}
            total=sum(exps.values())
            probs={k:round(v/total,5) for k,v in exps.items()}
            return max(probs,key=probs.get),probs

    def teach(self,source,target):
        with self.lock:
            self.add_node(source);self.add_node(target)
            predicted, before=self.predict(source)
            # Delta rule on a softmax categorical prediction.
            for candidate,p in before.items():
                key=self.key(source,candidate)
                delta=self.learning_rate*((1 if candidate==target else 0)-p)
                self.edges[key]=max(-4,min(4,self.edges.get(key,0)+delta))
            self.nodes[source]["visits"]+=1
            self.nodes[source]["activation"]=1.0
            self.nodes[target]["activation"]=min(1.0,self.nodes[target]["activation"]+0.5)
            for name,node in self.nodes.items():
                if name not in (source,target): node["activation"]*=0.85
            self.steps+=1
            after, probabilities=self.predict(source)
            event={"step":self.steps,"source":source,"target":target,"prediction_before":predicted,
                   "correct_before":predicted==target,"prediction_after":after,
                   "confidence_after":probabilities.get(target,0)}
            self.history.append(event)
            self.history=self.history[-500:]
            self.last_input=source
            self.save()
            return event

    def tick(self):
        with self.lock:
            old={k:v["activation"] for k,v in self.nodes.items()}
            new={}
            for target in self.nodes:
                influence=sum(math.tanh(w)*old[source] for key,w in self.edges.items()
                              for source,dest in [key.split("\u2192",1)] if dest==target)
                new[target]=max(0,min(1,0.92*old[target]+0.08*math.tanh(max(0,influence))))
            for name,value in new.items():self.nodes[name]["activation"]=value

    def snapshot(self):
        with self.lock:
            return {"nodes":[{"id":k,**v} for k,v in self.nodes.items()],
                    "edges":[{"source":k.split("\u2192",1)[0],"target":k.split("\u2192",1)[1],"weight":v}
                             for k,v in self.edges.items() if abs(v)>0.0001],
                    "steps":self.steps,"history":self.history[-30:]}

    def save(self):
        with self.lock:
            data={"nodes":self.nodes,"edges":self.edges,"history":self.history,"steps":self.steps}
            tmp=self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data,indent=2))
            os.replace(tmp,self.path)

    def load(self):
        with self.lock:
            data=json.loads(self.path.read_text())
            self.nodes=data["nodes"];self.edges=data["edges"]
            self.history=data.get("history",[]);self.steps=data.get("steps",0)
