classdef AuthorJournalExperiments
    properties
        authJournal;
    end

    properties
        path='/v/filer4b/v20q001/vvasuki/vishvas/work/graphTheory/+linkPrediction/';
        siamDataPath='/u/wtang/matlab/SocialNetwork/siam/'
        authJournalFile=strcat(path,'authJournal.mat');
    end
    
methods
    function predictor = Predictor()
        path='/v/filer4b/v20q001/vvasuki/vishvas/work/graphTheory/+linkPrediction/';
        authJournalFile=strcat(path,'authJournal.mat');
        predictor.authJournal = load(authJournalFile);
        predictor.authJournal = predictor.authJournal.authJournal;
    end

    function journalNW = lpWithAffiliationNw(this)
        %  Eventually, there will be a link prediction method which utilizes the affiliation network here. Until then, this will be dummy code to call functions written below.
        %  authNw_Journal = authJournal*authJournal';
        %  authNw_doc = authdoc*authdoc';
        this.checkJournalClustering(this.authJournal);
        journalNW =  10000;
    end

    function findCorrelation(authNw_Journal, authNw_doc)
        %  Find how correlated two adjascency matreces are.
        %  Incomplete. Do not know how to find correlation between matreces.
    end

    function checkJournalClustering(this, authJournal)
        %  Explore the way the journals cluster together
        journalNW = authJournal'*authJournal;
        journalNW = journalNW./(sum(sum(journalNW)));
        %  Couldn't use Hyuk Cho's C++ implementation: don't know dense matrix format.
        %  Observed that icc produces symmetric row and column clusterings once the cluster indeces are properly mapped. So, simply using the rowClusterIndices part of the output of icc.
        rowClusterIndices = icc(journalNW, 3, 3);
        %  Incomplete: Should build co-clustering matrix.
        
    end

    function make_authJournal
        %  Load author vs document and document vs journal matreces, create author vs journal matrix, then store this matrix in a file.
        load multiway_authdoc;
        load multiway_journal_labels;
        %  B_journal_labels ranges from 1 to 12.
        [numAuthors, numDocs] = size(authdoc);
        numJournals = max(B_journal_labels, [], 2);
        authJournal=zeros(numAuthors, numJournals);
        for auth=1:numAuthors
            docs = find(authdoc(auth,:));
            for doc = docs
                journal = B_journal_labels(doc);
                authJournal(auth, journal) = authJournal(auth, journal) + 1;
            end
        end
        %  Normalize the author vs journal matrix.
        authJournal=normr(authJournal);
        save authJournalFile, authJournal;
    end

end
end