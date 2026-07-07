package au.unimelb.evaluation;

import au.qut.apromore.importer.ImportEventLog;
import au.qut.apromore.importer.ImportProcessModel;
import org.deckfour.xes.model.XLog;
import org.processmining.antialignments.ilp.antialignment.AntiAlignmentParameters;
import org.processmining.antialignments.ilp.antialignment.AntiAlignmentValues;
import org.processmining.antialignments.ilp.antialignment.HeuristicAntiAlignmentAlgorithm;
import org.processmining.antialignments.ilp.util.AntiAlignments;
import org.processmining.framework.plugin.Progress;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.petrinet.replayresult.PNRepResult;

public class M4Progress {
    static long t0 = System.currentTimeMillis();
    static String sec(){ return String.format("%.1f",(System.currentTimeMillis()-t0)/1000.0); }
    static void log(String s){ System.err.println("[t="+sec()+"s] "+s); System.err.flush(); }

    static class LogProgress implements Progress {
        int min=0,max=0,val=0; String caption=""; long last=0;
        public void setMinimum(int m){min=m;}
        public void setMaximum(int m){max=m; log("PROGRESS set max="+m+" ("+caption+")");}
        public void setValue(int v){val=v; report();}
        public void setCaption(String c){caption=c; log("PHASE: "+c);}
        public String getCaption(){return caption;}
        public int getValue(){return val;}
        public void inc(){val++; report();}
        public void setIndeterminate(boolean b){}
        public boolean isIndeterminate(){return false;}
        public int getMinimum(){return min;}
        public int getMaximum(){return max;}
        public boolean isCancelled(){return false;}
        public void cancel(){}
        void report(){
            long now=System.currentTimeMillis();
            if(now-last>2000 || (max>0 && val>=max)){
                last=now;
                String pct = max>0 ? String.format("%.1f%%",100.0*val/max) : "?";
                log("PROGRESS "+val+"/"+max+" = "+pct+" ("+caption+")");
            }
        }
    }

    public static void main(String[] args) throws Exception {
        String path=args[0], logf=args[1], modelf=args[2];
        log("loading model "+path+modelf);
        ImportProcessModel ipm = new ImportProcessModel();
        Object[] nm = ipm.importPetriNetAndMarking(path+modelf);
        Petrinet pnet=(Petrinet)nm[0]; Marking im=(Marking)nm[1], fm=(Marking)nm[2];
        log("loading log "+path+logf);
        XLog xLog = new ImportEventLog().importEventLog(path+logf);
        log("STAGE 1/3: computing alignments (log size="+xLog.size()+")");
        AlignmentSetup as = new AlignmentSetup(pnet, xLog);
        PNRepResult alignments = as.getAlignment(null, im, fm, false, 0);
        log("STAGE 2/3: alignments done; computing anti-alignments");
        HeuristicAntiAlignmentAlgorithm alg = new HeuristicAntiAlignmentAlgorithm(pnet, im, fm, xLog, alignments, as.mapping);
        AntiAlignments aa = alg.computeAntiAlignments(new LogProgress(), new AntiAlignmentParameters(5, 1.0, 2, 2.0));
        log("STAGE 3/3: anti-alignments done; computing precision+generalization");
        AntiAlignmentValues v = alg.computePrecisionAndGeneralization(aa);
        log("DONE generalization="+v.getGeneralization());
        System.out.println("RESULT,"+logf+","+modelf+","+v.getGeneralization());
    }
}
